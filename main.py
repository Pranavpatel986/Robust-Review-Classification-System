import os
import json
import time
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
from monitor import SelfHealingMonitor  

# 1. Setup and Configuration
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Initializing with the verified threshold
monitor = SelfHealingMonitor(threshold=0.2)

MODEL_ID = "models/gemini-2.5-flash" 

TAXONOMY = [
    "Accessibility", "Fragrance", "Brand Value", "Feel / Finish", "Price", 
    "Product Safety", "Packaging", "Cleansing", "Efficacy", "Product Texture", 
    "Skin Care", "Companion Approval", "Convenience", "Brand Accountability", 
    "Skin Texture Improvement", "Brand For Me"
]

def classify_review(review_text):
    """
    Two-stage Sentiment-Anchored Reflexion loop with Few-Shot Grounding.
    """
    
    # INDENTED Correctly inside the function
    initial_prompt = f"""
    You are a Senior Consumer Insights Data Scientist. Perform a high-precision Semantic Audit.

    ### OPERATIONAL DEFINITIONS:
    1. Efficacy: Functional performance (lather, cleansing).
    2. Skin Care: Moisturizing, soothing, dermatological benefits.
    3. Brand For Me: Personal identification/loyalty.
    4. Companion Approval: Family/husband/others using it.

    Taxonomy: {TAXONOMY}

    ### FEW-SHOT EXAMPLES:
    - Review: "My husband loves how it smells, and it doesn't leave his skin dry." 
      -> Labels: ["Companion Approval", "Fragrance", "Skin Care"]
    - Review: "The pump broke on day one. Total waste of money." 
      -> Labels: ["Packaging", "Price"]

    Review: "{review_text}"

    Return ONLY JSON: {{"audit": "Extract phrases for labels", "factors": ["Label1"]}}
    """
    
    reflexion_prompt_template = """
    Review: "{review_text}"
    Audit: {audit}
    Proposed Labels: {initial_labels}

    Task: Verify labels against taxonomy: {taxonomy_list}
    Rule: Every label MUST have a direct quote in the audit. Remove any "guessed" labels.
    
    Return ONLY JSON: {{"factors": ["VerifiedLabel1"]}}
    """

    for attempt in range(7): 
        try:
            # Stage 1: Audit
            res1 = client.models.generate_content(
                model=MODEL_ID,
                contents=initial_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json" 
                )
            )
            data1 = json.loads(res1.text)
            
            # Stage 2: Reflexion
            res2 = client.models.generate_content(
                model=MODEL_ID,
                contents=reflexion_prompt_template.format(
                    review_text=review_text, 
                    audit=data1.get("audit", ""),
                    initial_labels=data1.get("factors", []),
                    taxonomy_list=TAXONOMY
                ),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json" 
                )
            )
            
            final_data = json.loads(res2.text)
            return final_data if isinstance(final_data, dict) else {"factors": final_data}
            
        except Exception as e:
            wait_time = 2 ** (attempt + 1) 
            print(f"Retrying due to: {e}. Sleeping {wait_time}s...")
            time.sleep(wait_time)
            
    return {"factors": []}

def run_pipeline():
    input_file = "bodywash-test.xlsx" 
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: {input_file} not found!")
        return

    print(f"--- Starting Final Pipeline (Optimized Few-Shot Mode) ---")
    
    test_df = pd.read_excel(input_file)
    test_df.columns = test_df.columns.str.strip()
    
    results = []
    for i, review in enumerate(test_df['Core Item']):
        if pd.isna(review):
            results.append("")
            continue
            
        clean_review = str(review).strip().replace("\n", " ")
        time.sleep(2) 
        
        prediction = classify_review(clean_review)
        factors = prediction.get('factors', []) if isinstance(prediction, dict) else []
        
        monitor.record_prediction(factors)
        valid_factors = [f for f in factors if f in TAXONOMY]
        results.append(", ".join(valid_factors))
        
        if (i + 1) % 5 == 0:
            print(f"Progress: {i + 1}/{len(test_df)} reviews analyzed.")

    test_df['Level 1 Factors'] = results
    test_df.to_csv("bodywash_test_final_optimized.csv", index=False)
    print(f"\nSUCCESS! Results saved to: bodywash_test_final_optimized.csv")

if __name__ == "__main__":
    run_pipeline()