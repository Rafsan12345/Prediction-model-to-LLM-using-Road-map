
# Sensor Prediction Agent + Small LLM

## 1. Project Overview

This project combines two different AI models:

1. **Custom Sensor Model (`sensor_model.h5`)**
   - Predicts `NORMAL` or `WARNING` from temperature and humidity.
   - Produces a prediction score.

2. **Small Language Model (Qwen2.5-0.5B-Instruct)**
   - Receives sensor data and the prediction from the custom model.
   - Generates a natural-language explanation in Bengali.

A Python agent connects these two models.

**Important:** The `.h5` model performs sensor prediction. The LLM explains the supplied prediction. The LLM does not automatically know the internal weights or exact decision rules of the `.h5` model.

---

## 2. System Architecture

```text
                 USER
                   |
                   v
       Temperature + Humidity
                   |
                   v
          Python Agent
          small_llm.py
                   |
                   v
         Input Preprocessing
         preprocessing.npz
             mean, std
                   |
                   v
        Custom Neural Network
           sensor_model.h5
                   |
                   v
          Prediction Score
            Example: 0.9577
                   |
                   v
         Threshold Decision
           Score >= 0.5?
             /       \
            /         \
        WARNING      NORMAL
            \         /
             \       /
                   |
                   v
         Python Prompt Builder
                   |
                   v
        Qwen2.5-0.5B-Instruct
             Small LLM
                   |
                   v
        Bengali Explanation
                   |
                   v
                  USER
```

### Main Components

| Component | Responsibility |
|---|---|
| User input | Provides temperature and humidity |
| Preprocessing | Normalizes input using saved statistics |
| `sensor_model.h5` | Predicts a sensor score |
| Threshold logic | Converts the score into a label |
| Python agent | Connects the prediction model and LLM |
| Qwen LLM | Generates a Bengali explanation |
| User interface | Displays the result in CMD |

---

## 3. Project Directory Structure

```text
fake_model_agent_project/
|
|-- train_model.py
|-- agent.py
|-- small_llm.py
|-- requirements.txt
|-- sensor_model.h5
|-- preprocessing.npz
|-- README.md
|-- agent_log.txt
```

### File Descriptions

| File | Purpose |
|---|---|
| `train_model.py` | Generates synthetic data and trains the neural network |
| `sensor_model.h5` | Stores the trained Keras model |
| `preprocessing.npz` | Stores input normalization statistics |
| `agent.py` | Runs the basic sensor prediction agent |
| `small_llm.py` | Connects the sensor model with Qwen |
| `requirements.txt` | Lists Python dependencies |
| `README.md` | Documents the project |
| `agent_log.txt` | May store logs if logging is implemented |

---

## 4. How the Sensor Model Was Trained

This demonstration uses synthetic sensor data rather than real-world measurements.

The artificial labels are generated using this rule:

```python
WARNING = (temperature >= 35) or (humidity >= 85)
NORMAL = otherwise
```

This means:

- If temperature is at least 35°C, the artificial label is `WARNING`.
- If humidity is at least 85%, the artificial label is `WARNING`.
- Otherwise, the artificial label is `NORMAL`.

A neural network is trained to learn patterns from these examples.

The model produces a score. The Python agent applies this decision threshold:

```python
prediction = "WARNING" if score >= 0.5 else "NORMAL"
```

The neural network may approximate the artificial labeling rule rather than reproduce it perfectly.

**Important:** This is a teaching demonstration, not a validated environmental safety model.

---

## 5. Role of Each AI Model

### Model 1: Custom Neural Network

File: `sensor_model.h5`

Responsibilities:

- Accepts temperature and humidity.
- Uses the same preprocessing method used during training.
- Produces a prediction score.
- Supplies the output to the Python agent.

Example:

```text
Temperature: 40°C
Humidity: 60%

Prediction score: 0.9577
Prediction label: WARNING
```

The score above is an example from a previous run, not a guaranteed result for every model version.

### Model 2: Qwen LLM

Model ID:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Responsibilities:

- Understands the supplied text prompt.
- Uses its pretrained language capabilities.
- Generates a Bengali explanation from the information provided.

The LLM is not retrained on the sensor dataset in this project.

### How They Work Together

The Python agent sends the LLM:

- Temperature
- Humidity
- Prediction score
- Prediction label
- Known artificial labeling rule

The LLM uses this information to explain the result.

It does not independently inspect the `.h5` model's internal weights or discover the training rule automatically.

---

## 6. Requirements

Required software:

- Windows 10 or Windows 11
- Python compatible with the installed TensorFlow and PyTorch versions
- Internet connection for the initial Qwen download
- Sufficient RAM and disk space

The first Qwen download may be large and can take time depending on internet speed.

---

## 7. Complete Windows CMD Workflow

### Step 1: Open CMD

Navigate to the project directory:

```cmd
cd /d "C:\Users\uiu\Downloads\fake_model_agent_project\fake_model_agent_project"
```

If your project is stored elsewhere, change the path.

### Step 2: Verify the Required Files

```cmd
dir
dir sensor_model.h5 preprocessing.npz
dir small_llm.py
```

Confirm that the required files exist.

### Step 3: Create a Virtual Environment

If Python 3.12 is installed:

```cmd
py -3.12 -m venv venv
```

Activate it:

```cmd
venv\Scripts\activate
```

If Python 3.12 is unavailable, check installed Python versions:

```cmd
py --list
```

Use a Python version compatible with the TensorFlow and PyTorch versions you install.

### Step 4: Install Dependencies

```cmd
python -m pip install --upgrade pip
pip install tensorflow numpy transformers torch
```

Alternatively, if `requirements.txt` contains all required packages:

```cmd
pip install -r requirements.txt
```

### Step 5: Train the Sensor Model (Optional)

If the existing model files are correct, retraining is unnecessary.

To train again:

```cmd
python train_model.py
```

This may overwrite existing model and preprocessing files. Back them up first if you need to preserve the current version.

Expected files:

```text
sensor_model.h5
preprocessing.npz
```

### Step 6: Run the LLM-Connected Agent

```cmd
python small_llm.py
```

The program loads the custom sensor model and the Qwen language model.

On the first run, Qwen may need to download its files.

### Step 7: Enter Sensor Values

When prompted:

```text
Temperature,Humidity (e.g. 40,60) or q to quit:
```

Enter:

```text
40,60
```

You can test another input:

```text
25,60
```

To exit:

```text
q
```

---

## 8. Expected Workflow

For an input such as:

```text
40,60
```

The program performs these operations:

1. Reads temperature and humidity.
2. Normalizes the input.
3. Runs the custom neural network.
4. Obtains the prediction score.
5. Converts the score into `NORMAL` or `WARNING`.
6. Builds a prompt containing the sensor values and prediction.
7. Sends the prompt to Qwen.
8. Prints the generated explanation.

---

## 9. Possible Output

The following is an illustrative example. Actual scores and generated text may differ.

```text
Loading sensor model...
Loading Qwen LLM...

Both models are loaded!

Temperature,Humidity (e.g. 40,60) or q to quit: 40,60

Your model score: 0.9577
Your model prediction: WARNING

--- Qwen's Bengali explanation ---
আপনার sensor model WARNING predict করেছে।
প্রদত্ত training rule অনুযায়ী তাপমাত্রা 35°C
বা বেশি হলে WARNING label দেওয়া হয়।
এখানে তাপমাত্রা 40°C।
```

The generated Bengali text may not always be grammatically correct because this is a small language model.

For another input:

```text
Temperature,Humidity (e.g. 40,60) or q to quit: 25,60

Your model score: <actual model score>
Your model prediction: <actual prediction>
```

The actual output must be obtained by running the program.

---

## 10. Understanding the Prediction Score

Example:

```text
Your model score: 0.9577
Your model prediction: WARNING
```

In this project:

```python
score >= 0.5
```

produces the `WARNING` label.

A score of `0.9577` is the neural network's output score. It does **not** establish that a real-world hazard has a 95.77% probability.

The model was trained using synthetic data and has not been validated for real environmental safety.

---

## 11. Troubleshooting

### Problem A: Model File Not Found

Check the files:

```cmd
dir sensor_model.h5
dir preprocessing.npz
```

Ensure that CMD is in the correct project directory.

### Problem B: Missing Python Package

For example:

```text
ModuleNotFoundError: No module named 'transformers'
```

Install the required package:

```cmd
python -m pip install transformers
```

For TensorFlow:

```cmd
python -m pip install tensorflow
```

### Problem C: Qwen Takes Too Long

A small language model can still generate slowly on a CPU.

Reduce `max_new_tokens` in `small_llm.py` to request a shorter answer.

### Problem D: Bengali Output Is Incoherent

A small LLM may have limited Bengali generation quality.

Try:

- Shorter prompts
- Clear instructions
- Shorter answers
- Providing the exact sensor data and prediction
- Providing the known artificial labeling rule

Prompt changes may help but cannot guarantee fluent Bengali.

### Problem E: Hugging Face Warning

Warnings about unauthenticated downloads or Windows symlinks do not necessarily indicate failure. Public model downloads can often work without authentication, and caching may use more disk space when symlinks are unavailable.

### Problem F: TensorFlow oneDNN Message

Messages about oneDNN or CPU instruction optimization are generally informational, not fatal errors.

---

## 12. Is This a Fully Autonomous AI Agent?

No. This is a basic Python agent that coordinates two models.

Current capabilities:

- Accepts sensor values from the user
- Performs neural network inference
- Applies a threshold
- Sends the result to an LLM
- Displays a Bengali explanation

Not yet implemented:

- Automatic live sensor readings
- MQTT, serial, or HTTP integration
- Tool calling
- Long-term memory
- Autonomous multi-step planning
- Safety-validated automatic control

---

## 13. Future Improvements

Possible next steps:

1. Read live sensor data from an ESP32.
2. Receive readings through MQTT or HTTP.
3. Ask questions in Bengali.
4. Add structured logging with timestamps.
5. Validate input ranges.
6. Evaluate the neural network using real labeled data.
7. Measure accuracy, precision, recall, and confusion matrix.
8. Restrict the LLM to verified information.
9. Add safety checks before any real-world action.

---

## 14. Complete Architecture Summary

```text
Temperature + Humidity
          |
          v
    Python Agent
          |
          v
    Preprocessing
          |
          v
   sensor_model.h5
          |
          v
  Prediction Score
          |
          v
 NORMAL / WARNING
          |
          v
 Prompt with sensor data,
 prediction and known rule
          |
          v
 Qwen2.5-0.5B-Instruct
          |
          v
 Bengali Explanation
```

## Conclusion

The custom `.h5` neural network performs the sensor prediction.

The pretrained Qwen LLM generates a natural-language explanation using the information supplied by Python.

The Python agent connects the two models.

**This project is an educational demonstration using synthetic data. Do not use it directly for medical decisions, industrial control, or real safety monitoring without appropriate real-world data, validation, testing, and fail-safe engineering.**

# Sensor Prediction Agent + Small LLM

## 1. Project Overview

This project combines two different AI models:

1. **Custom Sensor Model (`sensor_model.h5`)**
   - Predicts `NORMAL` or `WARNING` from temperature and humidity.
   - Produces a prediction score.

2. **Small Language Model (Qwen2.5-0.5B-Instruct)**
   - Receives sensor data and the prediction from the custom model.
   - Generates a natural-language explanation in Bengali.

A Python agent connects these two models.

**Important:** The `.h5` model performs sensor prediction. The LLM explains the supplied prediction. The LLM does not automatically know the internal weights or exact decision rules of the `.h5` model.

---

## 2. System Architecture

```text
                 USER
                   |
                   v
       Temperature + Humidity
                   |
                   v
          Python Agent
          small_llm.py
                   |
                   v
         Input Preprocessing
         preprocessing.npz
             mean, std
                   |
                   v
        Custom Neural Network
           sensor_model.h5
                   |
                   v
          Prediction Score
            Example: 0.9577
                   |
                   v
         Threshold Decision
           Score >= 0.5?
             /       \
            /         \
        WARNING      NORMAL
            \         /
             \       /
                   |
                   v
         Python Prompt Builder
                   |
                   v
        Qwen2.5-0.5B-Instruct
             Small LLM
                   |
                   v
        Bengali Explanation
                   |
                   v
                  USER
```

### Main Components

| Component | Responsibility |
|---|---|
| User input | Provides temperature and humidity |
| Preprocessing | Normalizes input using saved statistics |
| `sensor_model.h5` | Predicts a sensor score |
| Threshold logic | Converts the score into a label |
| Python agent | Connects the prediction model and LLM |
| Qwen LLM | Generates a Bengali explanation |
| User interface | Displays the result in CMD |

---

## 3. Project Directory Structure

```text
fake_model_agent_project/
|
|-- train_model.py
|-- agent.py
|-- small_llm.py
|-- requirements.txt
|-- sensor_model.h5
|-- preprocessing.npz
|-- README.md
|-- agent_log.txt
```

### File Descriptions

| File | Purpose |
|---|---|
| `train_model.py` | Generates synthetic data and trains the neural network |
| `sensor_model.h5` | Stores the trained Keras model |
| `preprocessing.npz` | Stores input normalization statistics |
| `agent.py` | Runs the basic sensor prediction agent |
| `small_llm.py` | Connects the sensor model with Qwen |
| `requirements.txt` | Lists Python dependencies |
| `README.md` | Documents the project |
| `agent_log.txt` | May store logs if logging is implemented |

---

## 4. How the Sensor Model Was Trained

This demonstration uses synthetic sensor data rather than real-world measurements.

The artificial labels are generated using this rule:

```python
WARNING = (temperature >= 35) or (humidity >= 85)
NORMAL = otherwise
```

This means:

- If temperature is at least 35°C, the artificial label is `WARNING`.
- If humidity is at least 85%, the artificial label is `WARNING`.
- Otherwise, the artificial label is `NORMAL`.

A neural network is trained to learn patterns from these examples.

The model produces a score. The Python agent applies this decision threshold:

```python
prediction = "WARNING" if score >= 0.5 else "NORMAL"
```

The neural network may approximate the artificial labeling rule rather than reproduce it perfectly.

**Important:** This is a teaching demonstration, not a validated environmental safety model.

---

## 5. Role of Each AI Model

### Model 1: Custom Neural Network

File: `sensor_model.h5`

Responsibilities:

- Accepts temperature and humidity.
- Uses the same preprocessing method used during training.
- Produces a prediction score.
- Supplies the output to the Python agent.

Example:

```text
Temperature: 40°C
Humidity: 60%

Prediction score: 0.9577
Prediction label: WARNING
```

The score above is an example from a previous run, not a guaranteed result for every model version.

### Model 2: Qwen LLM

Model ID:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Responsibilities:

- Understands the supplied text prompt.
- Uses its pretrained language capabilities.
- Generates a Bengali explanation from the information provided.

The LLM is not retrained on the sensor dataset in this project.

### How They Work Together

The Python agent sends the LLM:

- Temperature
- Humidity
- Prediction score
- Prediction label
- Known artificial labeling rule

The LLM uses this information to explain the result.

It does not independently inspect the `.h5` model's internal weights or discover the training rule automatically.

---

## 6. Requirements

Required software:

- Windows 10 or Windows 11
- Python compatible with the installed TensorFlow and PyTorch versions
- Internet connection for the initial Qwen download
- Sufficient RAM and disk space

The first Qwen download may be large and can take time depending on internet speed.

---

## 7. Complete Windows CMD Workflow

### Step 1: Open CMD

Navigate to the project directory:

```cmd
cd /d "C:\Users\uiu\Downloads\fake_model_agent_project\fake_model_agent_project"
```

If your project is stored elsewhere, change the path.

### Step 2: Verify the Required Files

```cmd
dir
dir sensor_model.h5 preprocessing.npz
dir small_llm.py
```

Confirm that the required files exist.

### Step 3: Create a Virtual Environment

If Python 3.12 is installed:

```cmd
py -3.12 -m venv venv
```

Activate it:

```cmd
venv\Scripts\activate
```

If Python 3.12 is unavailable, check installed Python versions:

```cmd
py --list
```

Use a Python version compatible with the TensorFlow and PyTorch versions you install.

### Step 4: Install Dependencies

```cmd
python -m pip install --upgrade pip
pip install tensorflow numpy transformers torch
```

Alternatively, if `requirements.txt` contains all required packages:

```cmd
pip install -r requirements.txt
```

### Step 5: Train the Sensor Model (Optional)

If the existing model files are correct, retraining is unnecessary.

To train again:

```cmd
python train_model.py
```

This may overwrite existing model and preprocessing files. Back them up first if you need to preserve the current version.

Expected files:

```text
sensor_model.h5
preprocessing.npz
```

### Step 6: Run the LLM-Connected Agent

```cmd
python small_llm.py
```

The program loads the custom sensor model and the Qwen language model.

On the first run, Qwen may need to download its files.

### Step 7: Enter Sensor Values

When prompted:

```text
Temperature,Humidity (e.g. 40,60) or q to quit:
```

Enter:

```text
40,60
```

You can test another input:

```text
25,60
```

To exit:

```text
q
```

---

## 8. Expected Workflow

For an input such as:

```text
40,60
```

The program performs these operations:

1. Reads temperature and humidity.
2. Normalizes the input.
3. Runs the custom neural network.
4. Obtains the prediction score.
5. Converts the score into `NORMAL` or `WARNING`.
6. Builds a prompt containing the sensor values and prediction.
7. Sends the prompt to Qwen.
8. Prints the generated explanation.

---

## 9. Possible Output

The following is an illustrative example. Actual scores and generated text may differ.

```text
Loading sensor model...
Loading Qwen LLM...

Both models are loaded!

Temperature,Humidity (e.g. 40,60) or q to quit: 40,60

Your model score: 0.9577
Your model prediction: WARNING

--- Qwen's Bengali explanation ---
আপনার sensor model WARNING predict করেছে।
প্রদত্ত training rule অনুযায়ী তাপমাত্রা 35°C
বা বেশি হলে WARNING label দেওয়া হয়।
এখানে তাপমাত্রা 40°C।
```

The generated Bengali text may not always be grammatically correct because this is a small language model.

For another input:

```text
Temperature,Humidity (e.g. 40,60) or q to quit: 25,60

Your model score: <actual model score>
Your model prediction: <actual prediction>
```

The actual output must be obtained by running the program.

---

## 10. Understanding the Prediction Score

Example:

```text
Your model score: 0.9577
Your model prediction: WARNING
```

In this project:

```python
score >= 0.5
```

produces the `WARNING` label.

A score of `0.9577` is the neural network's output score. It does **not** establish that a real-world hazard has a 95.77% probability.

The model was trained using synthetic data and has not been validated for real environmental safety.

---

## 11. Troubleshooting

### Problem A: Model File Not Found

Check the files:

```cmd
dir sensor_model.h5
dir preprocessing.npz
```

Ensure that CMD is in the correct project directory.

### Problem B: Missing Python Package

For example:

```text
ModuleNotFoundError: No module named 'transformers'
```

Install the required package:

```cmd
python -m pip install transformers
```

For TensorFlow:

```cmd
python -m pip install tensorflow
```

### Problem C: Qwen Takes Too Long

A small language model can still generate slowly on a CPU.

Reduce `max_new_tokens` in `small_llm.py` to request a shorter answer.

### Problem D: Bengali Output Is Incoherent

A small LLM may have limited Bengali generation quality.

Try:

- Shorter prompts
- Clear instructions
- Shorter answers
- Providing the exact sensor data and prediction
- Providing the known artificial labeling rule

Prompt changes may help but cannot guarantee fluent Bengali.

### Problem E: Hugging Face Warning

Warnings about unauthenticated downloads or Windows symlinks do not necessarily indicate failure. Public model downloads can often work without authentication, and caching may use more disk space when symlinks are unavailable.

### Problem F: TensorFlow oneDNN Message

Messages about oneDNN or CPU instruction optimization are generally informational, not fatal errors.

---

## 12. Is This a Fully Autonomous AI Agent?

No. This is a basic Python agent that coordinates two models.

Current capabilities:

- Accepts sensor values from the user
- Performs neural network inference
- Applies a threshold
- Sends the result to an LLM
- Displays a Bengali explanation

Not yet implemented:

- Automatic live sensor readings
- MQTT, serial, or HTTP integration
- Tool calling
- Long-term memory
- Autonomous multi-step planning
- Safety-validated automatic control

---

## 13. Future Improvements

Possible next steps:

1. Read live sensor data from an ESP32.
2. Receive readings through MQTT or HTTP.
3. Ask questions in Bengali.
4. Add structured logging with timestamps.
5. Validate input ranges.
6. Evaluate the neural network using real labeled data.
7. Measure accuracy, precision, recall, and confusion matrix.
8. Restrict the LLM to verified information.
9. Add safety checks before any real-world action.

---

## 14. Complete Architecture Summary

```text
Temperature + Humidity
          |
          v
    Python Agent
          |
          v
    Preprocessing
          |
          v
   sensor_model.h5
          |
          v
  Prediction Score
          |
          v
 NORMAL / WARNING
          |
          v
 Prompt with sensor data,
 prediction and known rule
          |
          v
 Qwen2.5-0.5B-Instruct
          |
          v
 Bengali Explanation
```

## Conclusion

The custom `.h5` neural network performs the sensor prediction.

The pretrained Qwen LLM generates a natural-language explanation using the information supplied by Python.

The Python agent connects the two models.

**This project is an educational demonstration using synthetic data. Do not use it directly for medical decisions, industrial control, or real safety monitoring without appropriate real-world data, validation, testing, and fail-safe engineering.**
