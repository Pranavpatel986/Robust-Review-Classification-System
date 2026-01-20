\# Bodywash NLP Classification Pipeline

An Project for multi-label consumer insight tagging.



\## How to Run

1\. `pip install -r requirements.txt`

2\. `python preprocess.py` (Aggregates training data)

3\. `python main.py` (Generates labeled test set)

4\. `python validate.py` (Verify accuracy)

"# Robust-Review-Classification-System" 


---
Here is a breakdown of how the system works and its key components:

## 1. The Core Innovation: "Reflexion" Loop

Unlike a simple AI prompt that might guess a category once, your `main.py` uses a **two-stage verification process**:

* **Stage 1 (Audit):** The AI acts as a Senior Analyst, first explaining the "Human Intent" and tone of the review.
* **Stage 2 (Reflexion):** The AI then looks at its own audit and the original text to double-check the labels, removing any that aren't explicitly supported. This significantly reduces "AI hallucinations."

## 2. File Architecture

The project is structured as a professional data science pipeline:

| File | Purpose |
| --- | --- |
| **`preprocess.py`** | Cleans the raw Excel data. It handles encoding issues (ASCII) and aggregates multiple tags for the same review into a single training row (`aggregated_train.csv`). |
| **`main.py`** | The "engine." It connects to the Gemini API, manages the Reflexion loop, handles rate-limiting with retries, and generates the final labels for the test set. |
| **`validate.py`** | The "judge." It compares the AI's predictions against ground-truth data using professional metrics like **Micro F1-Score** and **Jaccard Index** (overlap quality). |
| **`monitor.py`** | A safety feature. It tracks "Concept Drift." If the AI starts failing to generate tags for 20% of the recent reviews, it triggers an alert. |

## 3. The Taxonomy (The 16 Pillars)

The system is trained to look for specific business insights, including:

* **Sensory:** Fragrance, Feel / Finish, Product Texture.
* **Value:** Price, Brand Value, Accessibility.
* **Performance:** Cleansing, Efficacy, Skin Care.
* **Social/Misc:** Companion Approval (e.g., "my wife loves the smell"), Packaging, Convenience.

## 4. Key Professional Features

* **Self-Healing:** The `SelfHealingMonitor` ensures that if the model's performance drops in a production environment, you know immediately.
* **Error Resilience:** The `classify_review` function includes **exponential backoff**, meaning if the API is busy, it waits progressively longer ( seconds) before trying again, rather than just crashing.
* **Strict JSON Mode:** It forces the AI to respond in JSON format, making it easy for the Python script to parse the results without manual cleaning.

---
