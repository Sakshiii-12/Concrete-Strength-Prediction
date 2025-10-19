# Add code to suppress warnings
# This must be done BEFORE importing TensorFlow

import os
import warnings

# Suppress TensorFlow startup messages (must be set before tensorflow import)
# 0 = all messages are logged (default)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress the specific scikit-learn UserWarning about feature names
warnings.filterwarnings('ignore', category=UserWarning)

# Now, the rest of the script
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the Saved Model and the Scaler
print("Loading the trained model...")

try:
    # Load the model without compiling it first to avoid potential errors.
    model = tf.keras.models.load_model('concrete_strength_model.h5', compile=False)
    # Now, manually re-compile the model.
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("Model loaded and re-compiled successfully!")
except (FileNotFoundError, OSError):
    print("Error: 'concrete_strength_model.h5' not found.")
    print("Please make sure you have run the training script first to create this file.")
    exit()

# IMPORTANT: You must use the SAME scaler that was used for training.
try:
    df = pd.read_excel('Concrete_data.xls')
    df.columns = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 
                  'coarse_agg', 'fine_agg', 'age', 'strength']
    X = df.drop('strength', axis=1)
    scaler = StandardScaler().fit(X) # Fit the scaler on the original features
    print("Scaler is ready.")
except FileNotFoundError:
    print("Error: 'Concrete_data.xls' not found. This data file is needed to set up the scaler.")
    exit()


# Create the Prediction Function
def predict_strength(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input, verbose=0)
    predicted_strength = prediction[0][0]
    
    print("\n" + "="*40)
    print(f"Predicted Concrete Strength: {predicted_strength:.2f} MPa")
    print("="*40)
    return predicted_strength

# Run an Interactive and User-Friendly Demo 
if __name__ == "__main__":
    print("\n--- Concrete Strength Predictor ---")
    
    prompts = [
        " 1. Cement (kg/m³):              ",
        " 2. Blast Furnace Slag (kg/m³):   ",
        " 3. Fly Ash (kg/m³):              ",
        " 4. Water (kg/m³):                ",
        " 5. Superplasticizer (kg/m³):     ",
        " 6. Coarse Aggregate (kg/m³):     ",
        " 7. Fine Aggregate (kg/m³):       ",
        " 8. Age (days):                   "
    ]
    
    example_values = [310, 145, 0, 192, 9, 978, 779, 28]

    print("\nPlease enter the 8 component values for a concrete mixture.")
    print("For components not in your mix, enter 0.")
    print("\nHere is an example input:")
    for i, p in enumerate(prompts):
        print(f"{p}{example_values[i]}")

    print("\nYou can enter the values on one line, separated by commas.")
    print("Example: 310, 145, 0, 192, 9, 978, 779, 28")
    print("\nType 'exit' to quit.")

    while True:
        user_input = input("\nEnter concrete components > ")
        if user_input.lower() == 'exit':
            break
        
        try:
            values = [float(v.strip()) for v in user_input.split(',')]
            if len(values) == 8:
                predict_strength(values)
            else:
                print(f" Error: Please enter exactly 8 values. You entered {len(values)}.")
        except ValueError:
            print(" Error: Invalid input. Please enter only numbers separated by commas.")
