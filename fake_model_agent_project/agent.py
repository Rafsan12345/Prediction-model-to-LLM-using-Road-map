import numpy as np
import tensorflow as tf
from datetime import datetime

MODEL_FILE = "sensor_model.h5"
PREPROCESS_FILE = "preprocessing.npz"

# Load the trained neural network and preprocessing settings
model = tf.keras.models.load_model(MODEL_FILE, compile=False)
stats = np.load(PREPROCESS_FILE)
mean = stats["mean"]
std = stats["std"]

def save_log(message):
    """Tool: save an event to a local log file."""
    with open("agent_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} | {message}\n")

def explain_result(temp, humidity, probability):
    """Agent policy: map model output to a simple action."""
    if probability >= 0.5:
        status = "WARNING"
        action = "Record warning and advise checking the sensor environment."
    else:
        status = "NORMAL"
        action = "Record normal reading."

    print("\n--- Agent result ---")
    print(f"Temperature: {temp:.1f} C")
    print(f"Humidity:    {humidity:.1f}%")
    print(f"Model warning score: {probability:.1%}")
    print(f"Status: {status}")
    print(f"Action: {action}")

    save_log(
        f"temp={temp:.1f}, humidity={humidity:.1f}, "
        f"score={probability:.4f}, status={status}, action={action}"
    )
    print("Log saved to agent_log.txt")
    return status

def run_agent(temp, humidity):
    # Validate user input
    if not  -10 <= temp <= 80:
        print("Temperature must be between -10 and 80 C for this demo.")
        return
    if not 0 <= humidity <= 100:
        print("Humidity must be between 0 and 100%.")
        return

    # Prepare the input exactly as during training
    x = np.array([[temp, humidity]], dtype=np.float32)
    x = (x - mean) / std

    # Ask the model for a prediction
    prediction = model.predict(x, verbose=0)[0][0]
    explain_result(temp, humidity, float(prediction))

print("Fake Sensor AI Agent")
print("Enter 'exit' to quit.")
print("Example: 25,60")
print("Demo thresholds: WARNING if temperature >= 35 C OR humidity >= 85%")
print("This is a teaching demo, not a real safety alarm.\n")

while True:
    raw = input("Temperature,Humidity > ").strip()
    if raw.lower() in ("exit", "quit"):
        print("Agent stopped.")
        break

    try:
        parts = [float(v.strip()) for v in raw.split(",")]
        if len(parts) != 2:
            raise ValueError
        run_agent(parts[0], parts[1])
    except ValueError:
        print("Invalid input. Enter two numbers like: 25,60\n")
