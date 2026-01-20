# Bodywash NLP Classification Pipeline

A Project for multi-label consumer insight tagging.

## How to Run

1. `pip install -r requirements.txt`

2. `python preprocess.py` (Aggregates training data)

3. `python main.py` (Generates labeled test set)

4. `python validate.py` (Verify accuracy)

# Robust-Review-Classification-System

Here is a breakdown of how the system works and its key components:

## 1. The Core Innovation: "Reflexion" Loop

Unlike a simple AI prompt that might guess a category once, `main.py` uses a **two-stage verification process**:

- **Stage 1 (Audit):** The AI acts as a Senior Analyst, first explaining the "Human Intent" and tone of the review.
- **Stage 2 (Reflexion):** The AI then looks at its own audit and the original text to double-check the labels, removing any that aren't explicitly supported. This significantly reduces "AI hallucinations."

## 2. File Architecture

The project is structured as a professional data science pipeline:

| File            | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| **`preprocess.py`** | Cleans the raw Excel data. It handles encoding issues (ASCII) and aggregates multiple tags for the same review into a single training row (`aggregated_train.csv`). |
| **`main.py`**   | The "engine." It connects to the Gemini API, manages the Reflexion loop, handles rate-limiting with retries, and generates the final labels for the test set. |
| **`validate.py`** | The "judge." It compares the AI's predictions against ground-truth data using professional metrics like **Micro F1-Score** and **Jaccard Index** (overlap quality). |
| **`monitor.py`** | A safety feature. It tracks "Concept Drift." If the AI starts failing to generate tags for 20% of the recent reviews, it triggers an alert. |

## 3. The Taxonomy (The 16 Pillars)

The system is trained to look for specific business insights, including:

- **Sensory:** Fragrance, Feel / Finish, Product Texture.
- **Value:** Price, Brand Value, Accessibility.
- **Performance:** Cleansing, Efficacy, Skin Care.
- **Social/Misc:** Companion Approval (e.g., "my wife loves the smell"), Packaging, Convenience.

## 4. Key Professional Features

- **Self-Healing:** The `SelfHealingMonitor` ensures that if the model's performance drops in a production environment, you know immediately.
- **Error Resilience:** The `classify_review` function includes **exponential backoff**, meaning if the API is busy, it waits progressively longer (seconds) before trying again, rather than just crashing.
- **Strict JSON Mode:** It forces the AI to respond in JSON format, making it easy for the Python script to parse the results without manual cleaning.

### Data Collection (The Source)

- **The Input:** We have two primary Excel files: `bodywash-train.xlsx` (the teacher) and `bodywash-test.xlsx` (the student).
- **The Nature of Data:** These are consumer reviews—unstructured text written by humans. They are "messy" because people use slang, emojis, and varied sentence structures.
- **The Labels:** The collection process includes a "Ground Truth" (the `Level 1 Factors`), which are the tags previously assigned by human experts to show the AI what a "correct" answer looks like.

### Software Requirements (`requirements.txt`)

To process this data and connect to the Gemini AI, your environment must have specific "tools" (libraries) installed:

- **Data Handling (`pandas`, `openpyxl`):** These allow Python to read, filter, and group the Excel files. Without them, the code cannot "see" the reviews.
- **AI Engine (`google-genai`):** This is the bridge to the Gemini 2.5 Flash model. It allows your script to send a review to Google's servers and receive an analysis back.
- **Evaluation Metrics (`scikit-learn`, `numpy`):** These are used for the "Validation" step. They provide the mathematical formulas to calculate how accurate the AI is (F1-score, Hamming Loss).
- **Environment Safety (`python-dotenv`):** This is used to store your `GEMINI_API_KEY` securely. It ensures your private keys aren't hard-coded directly into the script, which is a professional security standard.

### Data Integrity Requirements

Before the "Preprocessing" starts, the data must meet certain "health" requirements:

- **Column Consistency:** The Excel files must have a column named `Core Item`.
- **Encoding:** The text must be readable (which is why your code later forces it into ASCII format).
- **Taxonomy Alignment:** The labels in the training data must match the 16 labels defined in your `TAXONOMY` list in `main.py`.

### Preprocessing Step (`preprocess.py`)

In this project, this logic is contained in `preprocess.py`. Here is a step-by-step explanation of the code:

#### 1. File Verification & Loading

The script first checks if the input file (`bodywash-train.xlsx`) exists. If it does, it loads it using `pandas`:

```python
df = pd.read_excel(input_file)
df.columns = df.columns.str.strip() # Removes accidental spaces in column names
```

- **Why?** Column names like `"Core Item "` (with a space) can cause code to crash. `str.strip()` ensures "Core Item" is always reachable.

#### 2. Data Cleaning (The "ASCII" Fix)

Raw reviews often contain emojis, curly quotes, or special characters that can confuse some CSV parsers or the AI.

```python
df = df.dropna(subset=['Core Item']) # Remove empty reviews
df['Core Item'] = df['Core Item'].str.encode('ascii', 'ignore').str.decode('ascii')
```

- **The Logic:** It converts the text to ASCII bytes and "ignores" any character it doesn't recognize (like a heart emoji or a weird smart quote), then converts it back to a string. This ensures the text is "clean" and standard.

#### 3. Multi-Label Aggregation (Grouping)

This is the most critical part of the script. In your raw Excel file, if a single review has three different tags (e.g., Fragrance, Price, and Packaging), it might appear as **three separate rows** with the same review text.

The code collapses these into one single row per review:

```python
aggregated = df.groupby('Core Item')['Level 1 Factors'].apply(lambda x: str(list(set(x)))).reset_index()
```

- **`groupby('Core Item')`**: Finds all rows where the review text is identical.
- **`set(x)`**: Removes duplicate tags for that specific review.
- **`list(...)`**: Collects the unique tags into a list.
- **`str(...)`**: Converts that list into a string (e.g., `['Fragrance', 'Price']`) so it can be saved in a CSV cell.

#### 4. Saving the Output

Finally, it saves the cleaned data into `aggregated_train.csv`.

```python
aggregated.to_csv("aggregated_train.csv", index=False)
```

This new file is what `validate.py` uses to check if the Gemini AI's predictions match the "Ground Truth" provided by humans.

### Validation Step (`validate.py`)

Before you use the AI to label thousands of new, unseen reviews, you must first "test the engine" using the data you just preprocessed. This step ensures that the Gemini 2.5 Flash model actually understands your 16-point taxonomy and isn't making random guesses.

Here is how the validation step works in your specific code:

#### 1. The "Health Check" Sample

The script doesn't test the entire training set (which would be expensive and slow). Instead, it takes a random sample:

```python
df = pd.read_csv("aggregated_train.csv").sample(10) 
```

It takes 10 reviews where we already know the correct human-labeled tags (the "Ground Truth").

#### 2. The Prediction vs. Truth Comparison

For each of those 10 reviews, the script:

1. Sends the review text to the `classify_review` function (the Gemini Reflexion loop).
2. Gets the AI's predicted labels.
3. Compares them to the human labels in `aggregated_train.csv`.

#### 3. Professional Performance Metrics

This is where the project shifts from "cool AI demo" to "professional engineering." The script calculates four key metrics to tell you if the model is ready:

- **Strict Accuracy (Exact Match):** Did the AI get *every single* tag right for a review? (Very hard to achieve).
- **Soft Accuracy (Partial Match):** Did the AI get at least *one* tag right? (Good for catching the general theme).
- **Jaccard Index (Average Overlap):** This measures the intersection over union. If the human said `[Fragrance, Price]` and the AI said `[Fragrance, Packaging]`, the Jaccard index calculates the 50% overlap.
- **Hamming Loss:** This tells you how many labels, on average, the AI is getting wrong (either missing a tag or adding an extra one).

#### 4. Why this step is vital

By running `validate.py`, you answer the question: **"Can I trust this AI's output?"**

- If your **Jaccard Index is > 70%**, your pipeline is robust and ready to run on the real test data.
- If the **Hamming Loss is high**, you know you need to go back and tweak your `initial_prompt` in `main.py` before wasting money/credits on the full dataset.

Once validation passes, you proceed to **Execution (`main.py`)** to label the actual `bodywash-test.xlsx` file.

### Pipeline Execution (`main.py`)

After validation, the definitive next step is **Pipeline Execution (`main.py`)**. This is the stage where the system moves from "testing" to "production," processing the actual unlabeled test data to generate business insights.

Here is an explanation of the key code blocks that handle this execution:

#### 1. The Loop and Rate Limiting

The code iterates through every review in your test file (`bodywash-test.xlsx`). Because AI APIs have limits on how many requests you can send per minute, the code includes a safety pause:

```python
for i, review in enumerate(test_df['Core Item']):
    # ... logic ...
    time.sleep(2) # Pause for 2 seconds to manage rate limits
```

#### 2. The Multi-Stage Reflexion Loop

This is the "brain" of the execution. Instead of a single prompt, it runs the `classify_review` function which performs a two-step "Reflexion":

- **Stage 1 (Audit):** The code asks Gemini to write an "Audit" first. This forces the AI to think about the *intent* (e.g., "The customer is complaining about the price but loves the smell").
- **Stage 2 (Refinement):** The code feeds that Audit back into Gemini, asking it to finalize the labels based *only* on that reasoning.

#### 3. Integrated Health Monitoring

While the labels are being generated, the `monitor.py` logic is running in the background. For every prediction made, the script sends the result to the `SelfHealingMonitor`:

```python
# Record to the monitor
monitor.record_prediction(factors)
```

If the AI starts returning empty lists (meaning it's confused or the data has changed significantly), the monitor will print a **"Concept Drift Detected"** alert. This prevents the pipeline from running to completion with bad data.

#### 4. Data Serialization and Export

Once the loop finishes, the code doesn't just print the results; it maps them back to the original structure and saves them as a new file.

```python
test_df['Level 1 Factors'] = results
output_file = "bodywash_test_flash_final.csv"
test_df.to_csv(output_file, index=False)
```

### Output & Insight Generation

The **Output & Insight Generation** stage is where the raw computational work of the AI is transformed into actionable business intelligence. While the previous steps focus on *how* to process the data, this stage focuses on the *value* derived from it.

In your project, this involves two distinct phases: **Data Finalization** and **Strategic Interpretation**.

#### 1. Data Finalization (The CSV Output)

Once the `main.py` execution loop finishes, the script compiles all predictions into a structured format.

- **Structure:** The script maps the AI’s "Reflexion" results back to the original test reviews.
- **Result:** It generates `bodywash_test_flash_final.csv`. This file acts as a "Categorized Master List" where every customer's voice is now tagged with one or more of your 16 taxonomy factors (e.g., *Fragrance*, *Product Safety*, or *Skin Texture Improvement*).

#### 2. Strategic Interpretation (The "Insight" Phase)

This is where you look at the "Big Picture." Instead of reading 500 individual reviews, you use the labels to answer critical business questions.

- **Volume Analysis:** You calculate the frequency of each tag. If *Packaging* appears in 30% of reviews but *Price* only appears in 5%, the company knows that fixing a leaky bottle is more urgent than offering a discount.
- **Sentiment Correlation:** By looking at which factors appear together, you can find hidden patterns. For example, if *Brand Value* always appears alongside *Fragrance*, it suggests that the scent of the product is the primary driver of brand loyalty.
- **Gap Identification:** If certain categories like *Brand Accountability* are never mentioned, it might mean customers don't currently associate the brand with social responsibility, or the AI needs better prompting to detect those subtle signals.

#### 3. The "So What?" Factor

The ultimate goal of this stage is to move from **Data** to **Decisions**.

- **Marketing:** "Our AI shows customers love the 'Feel/Finish'—let's use that language in our next ad campaign."
- **R&D:** "There are high 'Product Safety' mentions regarding skin irritation; we need to review our formula."
- **Retail:** "People are complaining about 'Accessibility' in Target stores; we need to check our distribution in those regions."