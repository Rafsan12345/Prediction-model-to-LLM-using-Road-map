
import numpy as np
import tensorflow as tf
from transformers import pipeline

# 1. Load your own trained sensor model
print("Loading sensor model...")
sensor_model = tf.keras.models.load_model(
    "sensor_model.h5",
    compile=False
)

# 2. Load the preprocessing information
stats = np.load("preprocessing.npz")
mean = stats["mean"]
std = stats["std"]

# 3. Load the small language model
print("Loading Qwen LLM...")
llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)

print("\nBoth models are loaded!")

while True:
    try:
        text = input(
            "\nTemperature,Humidity (e.g. 40,60) "
            "or q to quit: "
        ).strip()

        if text.lower() == "q":
            break

        temperature, humidity = map(
            float, text.split(",")
        )

        # 4. Prepare the sensor input
        sensor_input = np.array(
            [[temperature, humidity]],
            dtype=np.float32
        )

        normalized_input = (
            sensor_input - mean
        ) / std

        # 5. Get prediction from YOUR .h5 model
        score = float(
            sensor_model.predict(
                normalized_input,
                verbose=0
            )[0][0]
        )

        prediction = (
            "WARNING" if score >= 0.5
            else "NORMAL"
        )

        print(f"\nYour model score: {score:.4f}")
        print(f"Your model prediction: {prediction}")

        # 6. Send the real prediction to Qwen
        messages = [
            {
                "role": "system",
                "content": (
                    "তুমি একটি IoT assistant। "
                    "সহজ বাংলায় উত্তর দাও। "
                    "শুধু দেওয়া sensor data ও prediction "
                    "ব্যবহার করে ব্যাখ্যা করো। "
                    "prediction পরিবর্তন করবে না।"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Temperature: {temperature} C\n"
                    f"Humidity: {humidity}%\n"
                    f"আমার trained sensor model-এর "
                    f"score: {score:.4f}\n"
                    f"Prediction: {prediction}\n"
                    "এই ফলাফল সহজ বাংলায় ব্যাখ্যা করো।"
                )
            }
        ]

        result = llm(
            messages,
            max_new_tokens=120,
            do_sample=False
        )

        explanation = result[0]["generated_text"][-1]["content"]

        print("\n--- Qwen's Bengali explanation ---")
        print(explanation)

    except ValueError:
        print("ভুল input! উদাহরণ: 40,60")

    except Exception as error:
        print("Error:", error)

print("Agent বন্ধ হয়েছে।")
