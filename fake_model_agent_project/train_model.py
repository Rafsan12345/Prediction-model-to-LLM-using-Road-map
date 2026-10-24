import numpy as np
import tensorflow as tf

# Reproducible demo
np.random.seed(42)
tf.random.set_seed(42)

# Fake data: each row is [temperature_C, humidity_percent]
n = 3000
temperature = np.random.uniform(10, 50, n)
humidity = np.random.uniform(10, 100, n)
X = np.column_stack([temperature, humidity]).astype(np.float32)

# Artificial rule for labels:
# WARNING if temperature >= 35 C OR humidity >= 85%
y = ((temperature >= 35) | (humidity >= 85)).astype(np.float32)

# Normalize input features using training-data statistics
mean = X.mean(axis=0)
std = X.std(axis=0)
X_scaled = (X - mean) / std

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_scaled, y,
    epochs=12,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Save model in HDF5 format
model.save("sensor_model.h5")

# Save preprocessing statistics; agent must use the same values
np.savez("preprocessing.npz", mean=mean, std=std)

loss, accuracy = model.evaluate(X_scaled, y, verbose=0)
print(f"\nTraining accuracy: {accuracy:.3f}")
print("Saved: sensor_model.h5")
print("Saved: preprocessing.npz")
print("Labels are based on an artificial rule, not real safety data.")
