# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load and Prepare the Data 
# CORRECTED LINE: Use pd.read_excel to load your '.xls' file.
try:
    df = pd.read_excel('Concrete_Data.xls')
except FileNotFoundError:
    print("Error: 'Concrete_data.xls' not found. Make sure the file is in the same folder.")
    exit()

# It's good practice to clean and shorten the column names
df.columns = [
    'cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
    'coarse_agg', 'fine_agg', 'age', 'strength'
]

# Preprocessing
X = df.drop('strength', axis=1)
y = df['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the ANN Model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# Train the Model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    verbose=1
)

# Evaluate the Model and Visualize Results
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Mean Absolute Error (MAE): {mae:.2f} MPa")

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred)
plt.plot([0, 80], [0, 80], 'r--', linewidth=2)
plt.xlabel('Actual Concrete Strength (MPa)', fontsize=14)
plt.ylabel('Predicted Concrete Strength (MPa)', fontsize=14)
plt.title('Model Performance: Predicted vs. Actual', fontsize=16)
plt.grid(True)
plt.show()

# Save the Final Model
# This saves the complete model to a single file.
print("\n Saving the trained model to 'concrete_strength_model.h5'...")
model.save('concrete_strength_model.h5')
print("Model saved successfully!")