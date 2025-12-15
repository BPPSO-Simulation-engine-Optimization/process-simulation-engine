"""
Example: Save and load resource availability model.
Demonstrates how to train once, save, and reuse the model.
"""
import sys
sys.path.append('pm4py-release')

import time
from datetime import datetime
from pathlib import Path
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_dataframe
from resources.resource_availabilities import AdvancedResourceAvailabilityModel

LOG_FILE = r"Dataset\BPI Challenge 2017.xes"
MODEL_FILE = "models/bpic2017_resource_model.pkl"

print("="*80)
print("MODEL SAVE/LOAD DEMONSTRATION")
print("="*80)

# Load event log
print("\n[1/4] Loading BPIC 2017 dataset...")
log = xes_importer.apply(LOG_FILE)
df = convert_to_dataframe(log)
print(f"      Loaded {len(df):,} events")

# Check if model already exists
if Path(MODEL_FILE).exists():
    print(f"\n[2/4] Pre-trained model found: {MODEL_FILE}")
    print("      Loading model (fast)...")
    
    start_time = time.time()
    model = AdvancedResourceAvailabilityModel.load_model(MODEL_FILE, df)
    loading_time = time.time() - start_time
    
    print(f"      Loading completed in {loading_time:.2f} seconds")
else:
    print(f"\n[2/4] No pre-trained model found")
    print("      Training model from scratch (this will take 10-30 seconds)...")
    
    start_time = time.time()
    model = AdvancedResourceAvailabilityModel(
        df,
        enable_pattern_mining=True,
        enable_lifecycle_tracking=True
    )
    training_time = time.time() - start_time
    
    print(f"      Training completed in {training_time:.2f} seconds")
    
    # Save for future use
    print(f"\n[3/4] Saving model to {MODEL_FILE}...")
    model.save_model(MODEL_FILE)
    print("      Model saved for future use!")

# Use the model
print(f"\n[4/4] Using the model...")
test_resource = "User_5"
test_time = datetime(2016, 6, 15, 10, 0)

print(f"\nQuery: Is '{test_resource}' available at {test_time}?")
is_available = model.is_available(test_resource, test_time)
probability = model.predict_availability_probability(test_resource, test_time)

print(f"  Available: {is_available}")
print(f"  Probability: {probability:.1%}")

print("\n" + "="*80)
print("DONE")
print("="*80)
print(f"\nNext time you run this script, it will load from {MODEL_FILE}")
print("and be much faster (loading takes <1 second instead of 10-30 seconds)!")
