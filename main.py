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

# Gemini 2.5 Flash works best on the standard v1 endpoint
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Use the exact Flash model name from your system check
MODEL_ID = "models/gemini-2.5-flash" 

TAXONOMY = [
    "Accessibility", "Fragrance", "Brand Value", "Feel / Finish", "Price", 
    "Product Safety", "Packaging", "Cleansing", "Efficacy", "Product Texture", 
    "Skin Care", "Companion Approval", "Convenience", "Brand Accountability", 
    "Skin Texture Improvement", "Brand For Me"
]

def classify_review(review_text):
    """
    Two-stage Sentiment-Anchored Reflexion loop using Gemini 2.5 Flash.
    """
    
    initial_prompt = f"""
    You are a Senior Consumer Insights Analyst. Your goal is to map the HUMAN INTENT of a review.
    
    ### ANALYTICAL FRAMEWORK:
    1. **Sentiment Anchoring**: Identify the overall tone (Positive, Negative, or Mixed).
    2. **Intent Audit**: Resolve any ambiguous phrases based on that tone.
    3. **Behavioral Mapping**: Look for signals like household sharing or loyalty.

    Taxonomy: {TAXONOMY}

    ### TASK:
    Provide an 'Audit' of the intent, then list the factors.
    Review: "{review_text}"
    
    Return ONLY JSON: {{"audit": "Explain intent", "factors": ["Label1"]}}
    """
    
    reflexion_prompt_template = """
    Review: "{review_text}"
    Audit: {audit}
    Initial Labels: {initial_labels}

    Task:
    - Ensure labels align with the Audit.
    - Remove labels that lack explicit semantic support.
    
    Return ONLY JSON: {{"factors": ["Label1", "Label2"]}}
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
                    initial_labels=data1.get("factors", [])
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
            print(f"Server busy/Error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    return {"factors": []}

def run_pipeline():
    input_file = "bodywash-test.xlsx" 
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: {input_file} not found!")
        return

    print(f"--- Starting Final Pipeline (Gemini 2.5 Flash) ---")
    
    test_df = pd.read_excel(input_file)
    test_df.columns = test_df.columns.str.strip()
    
    results = []
    for i, review in enumerate(test_df['Core Item']):
        if pd.isna(review):
            results.append("")
            continue
            
        clean_review = str(review).strip().replace("\n", " ")
        
        # FLASH PAUSE: Flash has higher RPM, so 2 seconds is enough
        time.sleep(2) 
        
        prediction = classify_review(clean_review)
        factors = prediction.get('factors', []) if isinstance(prediction, dict) else []
        
        monitor.record_prediction(factors)
        valid_factors = [f for f in factors if f in TAXONOMY]
        results.append(", ".join(valid_factors))
        
        if (i + 1) % 5 == 0:
            print(f"Progress: {i + 1}/{len(test_df)} reviews processed.")

    test_df['Level 1 Factors'] = results
    output_file = "bodywash_test_flash_final.csv"
    test_df.to_csv(output_file, index=False)
    print(f"\nSUCCESS! Results saved to: {output_file}")

if __name__ == "__main__":
    run_pipeline()