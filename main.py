import json
import pandas as pd
import numpy as np
import time
import logging
import os
import paho.mqtt.client as mqtt
from datetime import datetime
from typing import Dict, Optional, Any
import joblib  # As requested
import tensorflow as tf  # As requested

# --- MQTT Configuration ---
# !!! MAKE THESE MATCH YOUR ESP32 CODE !!!
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_STATUS_TOPIC = "ac-meter-myname-123/status"
MQTT_COMMAND_TOPIC = "ac-meter-myname-123/command" 

# --- Local File Config ---
CSV_FILE = "energy_data_mqtt_dual.csv"

# --- Model file paths (for reference, but unused) ---
IFOREST_MODEL_FILE = "iforest_model.joblib"
IFOREST_SCALER_FILE = "iforest_scaler.joblib"
AUTOENCODER_MODEL_FILE = "autoencoder_model.h5"
TF_SCALER_FILE = "tf_scaler.joblib"
THRESHOLDS_FILE = "model_thresholds.json"

# Anomaly detection config
STD_DEVIATIONS = 3.0
MIN_POWER_ANOMALY = 500.0
MIN_DATA_POINTS = 20

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_monitor_mqtt.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Global Variables ---
live_data_df = pd.DataFrame()
client = None  # MQTT client
iforest_model = None
iforest_scaler = None
autoencoder_model = None
tf_scaler = None
model_thresholds = None

# --- Model Loading (OPTIONAL) ---
def load_models():
    """
    Loads pre-trained models.
    NOTE: These models MUST be trained on the new 12-feature
    (v1,c1,p1...v2,c2,p2...) data format to work.
    """
    global iforest_model, iforest_scaler, autoencoder_model, tf_scaler, model_thresholds
    try:
        # Check if all model files exist
        if not all(os.path.exists(f) for f in [IFOREST_MODEL_FILE, IFOREST_SCALER_FILE, AUTOENCODER_MODEL_FILE, TF_SCALER_FILE, THRESHOLDS_FILE]):
            logger.warning("One or more model files are missing. ML models will be skipped.")
            return False
            
        iforest_model = joblib.load(IFOREST_MODEL_FILE)
        iforest_scaler = joblib.load(IFOREST_SCALER_FILE)
        autoencoder_model = tf.keras.models.load_model(AUTOENCODER_MODEL_FILE)
        tf_scaler = joblib.load(TF_SCALER_FILE)
        with open(THRESHOLDS_FILE, 'r') as f:
            model_thresholds = json.load(f)
            
        logger.info("All models, scalers, and thresholds loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}. ML models will be skipped.")
        return False

# --- Data Validation ---
def validate_data(data: Dict) -> bool:
    """Validates data for BOTH meters received from MQTT."""
    # All fields are now required
    required_fields = [
        'v1', 'c1', 'p1', 'e1', 'f1', 'pf1',
        'v2', 'c2', 'p2', 'e2', 'f2', 'pf2'
    ]
    try:
        if not all(field in data for field in required_fields):
            logger.warning(f"Missing fields in data: {data.keys()}")
            return False
        for field in required_fields:
            float(data[field]) # Check if conversion to float works
        return True
    except (ValueError, TypeError):
        logger.warning(f"Invalid data types received from MQTT: {data}")
        return False

# --- Save Data to CSV ---
def save_to_csv(data: Dict, timestamp: pd.Timestamp):
    """Saves the DUAL-meter data to CSV"""
    try:
        new_row = {
            'timestamp': timestamp,
            'voltage1': float(data['v1']),
            'current1': float(data['c1']),
            'power1': float(data['p1']),
            'energy1': float(data['e1']),
            'frequency1': float(data['f1']),
            'pf1': float(data['pf1']),
            'voltage2': float(data['v2']),
            'current2': float(data['c2']),
            'power2': float(data['p2']),
            'energy2': float(data['e2']),
            'frequency2': float(data['f2']),
            'pf2': float(data['pf2'])
        }
        # Add time-based features
        new_row['hour'] = timestamp.hour
        new_row['dayofweek'] = timestamp.dayofweek
        new_row['is_weekday'] = (new_row['dayofweek'] < 5).astype(int)

        df = pd.DataFrame([new_row])
        df.set_index('timestamp', inplace=True)
        
        try:
            with open(CSV_FILE, 'a') as f:
                df.to_csv(f, index=True, header=not f.tell())
            logger.info(f"Data appended to {CSV_FILE}")
        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")
    except Exception as e:
        logger.error(f"Error preparing CSV data: {e}")

# --- Publish Control Command via MQTT ---
def send_control_command(apt: int, state: str):
    """Publishes a control command for a specific apartment."""
    if not client:
        logger.error("MQTT client not initialized. Cannot send command.")
        return
        
    try:
        payload = json.dumps({f"apt{apt}": state}) # Dynamic key: apt1 or apt2
        client.publish(MQTT_COMMAND_TOPIC, payload, qos=1)
        logger.info(f"Published control command for Apt {apt}: {state}")
    except Exception as e:
        logger.error(f"Error publishing MQTT command: {e}")

# --- Anomaly Detection ---
def run_anomaly_detection(data: Dict, timestamp: pd.Timestamp):
    global live_data_df
    
    try:
        new_row_data = {
            'voltage1': float(data['v1']),
            'current1': float(data['c1']),
            'power1': float(data['p1']),
            'voltage2': float(data['v2']),
            'current2': float(data['c2']),
            'power2': float(data['p2']),
            'hour': timestamp.hour,
            'dayofweek': timestamp.dayofweek,
            'is_weekday': (timestamp.dayofweek < 5).astype(int)
        }
        new_row = pd.Series(new_row_data, name=timestamp)
        live_data_df = pd.concat([live_data_df, new_row.to_frame().T])
    except Exception as e:
        logger.error(f"Failed to parse data for anomaly detection: {e}")
        return

    if len(live_data_df) > 1440:
        live_data_df = live_data_df.iloc[-1440:]
    
    # --- NOW WE CHECK BOTH APARTMENTS ---
    for apt in [1, 2]:
        power_col = f'power{apt}'
        current_power = new_row[power_col]
        
        # --- 1. Statistical Anomaly (THIS ONE WORKS) ---
        if current_power > MIN_POWER_ANOMALY:
            if len(live_data_df) > MIN_DATA_POINTS:
                mean = live_data_df[power_col].mean()
                std = live_data_df[power_col].std()
                if std > 0: 
                    threshold = mean + (std * STD_DEVIATIONS)
                    if current_power > threshold:
                        logger.warning(f"[Statistical] Anomaly for Apt {apt}: {current_power:.1f}W (Threshold: {threshold:.1f})")
                        send_control_command(apt, "OFF")
                        continue # Move to check the next apartment
                
        # --- 2. Isolation Forest Anomaly (STUBBED OUT) ---
        if iforest_model and iforest_scaler:
            try:
                # You must retrain your scaler and model on the new 12-feature CSV data format.
                # features_to_use = ['power1', 'current1', 'power2', 'current2', 'hour', ...]
                # ...
                # prediction = iforest_model.predict(...)
                # if prediction[0] == -1:
                #     logger.warning(f"[IsolationForest] Anomaly detected for Apt {apt}: {current_power:.1f}W")
                #     send_control_command(apt, "OFF")
                #     continue
                pass 
            except Exception as e:
                logger.error(f"Isolation Forest (Apt {apt}) prediction error: {e}")
                logger.error("This is likely due to a model/data mismatch. Retrain your model.")

        # --- 3. Autoencoder Anomaly (STUBBED OUT) ---
        if autoencoder_model and tf_scaler and model_thresholds:
            try:
                # You must retrain your scaler and model on the new 12-feature CSV data format.
                # tf_features = ['voltage1', 'current1', 'power1', 'voltage2', 'current2', 'power2']
                # ...
                # mae = ...
                # if mae[0] > model_thresholds['tf_mae_threshold']:
                #     logger.warning(f"[Autoencoder] Anomaly for Apt {apt}: {current_power:.1f}W (MAE: ...)")
                #     send_control_command(apt, "OFF")
                #     continue
                pass
            except Exception as e:
                logger.error(f"Autoencoder (Apt {apt}) prediction error: {e}")
                logger.error("This is likely due to a model/data mismatch. Retrain your model.")

# --- MQTT Callback Functions ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"Connected to MQTT Broker: {MQTT_BROKER}")
        client.subscribe(MQTT_STATUS_TOPIC)
        logger.info(f"Subscribed to topic: {MQTT_STATUS_TOPIC}")
    else:
        logger.error(f"Failed to connect to MQTT, return code {rc}")

def on_message(client, userdata, msg):
    logger.debug(f"Received message on topic {msg.topic}")
    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        
        if validate_data(data):
            timestamp = pd.Timestamp.now(tz='UTC')
            save_to_csv(data, timestamp)
            run_anomaly_detection(data, timestamp)
        else:
            logger.warning(f"Invalid data received from MQTT: {payload}")
            
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from MQTT: {msg.payload.decode('utf-8')}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")

# --- Main Script ---
def main():
    global client
    
    # Attempt to load models, but continue if they fail.
    load_models() 
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    logger.info(f"Connecting to MQTT broker {MQTT_BROKER}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()  # This blocks and handles all MQTT traffic
    except KeyboardInterrupt:
        logger.info("Stopping MQTT loop...")
        client.disconnect()
    except Exception as e:
        logger.error(f"Failed to connect to MQTT: {e}")

if __name__ == "__main__":
    main()
