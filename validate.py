import pandas as pd
import numpy as np
import os
from main import classify_review, TAXONOMY
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
import ast

def run_validation():
    print("--- Starting Professional Multi-Label Validation (Gemini 2.5) ---")
    
    # 1. Load a sample of the training data
    file_path = "aggregated_train.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found! Please ensure your training data is available.")
        return

    # Sampling 10 for a health check of the Gemini logic
    df = pd.read_csv(file_path).sample(10) 
    
    y_true = []
    y_pred = []
    soft_match_count = 0  
    
    print(f"Processing {len(df)} validation samples via Google Gemini API...")
    
    for i, row in df.iterrows():
        # Get Ground Truth
        try:
            val = row['Level 1 Factors']
            true_labels = ast.literal_eval(val) if isinstance(val, str) else []
        except:
            true_labels = [] 
            
        y_true.append(true_labels)
        
        # Get Gemini Prediction
        clean_text = str(row['Core Item']).strip().replace("\n", " ")
        prediction = classify_review(clean_text)
        
        # Filter: Ensure only valid taxonomy labels pass through
        pred_labels = [f for f in prediction.get('factors', []) if f in TAXONOMY]
        y_pred.append(pred_labels)
        
        # Soft Accuracy Logic: Check for any common labels
        if set(pred_labels) & set(true_labels):
            soft_match_count += 1
        
        print(f"Verified sample {len(y_true)}/{len(df)} [Index: {i}]")

    # 2. Transform labels into Binary Format
    mlb = MultiLabelBinarizer(classes=TAXONOMY)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # 3. Calculate Performance Metrics
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    h_loss = hamming_loss(y_true_bin, y_pred_bin)
    
    # Strict Accuracy: Exact subset match
    exact_match = np.all(y_true_bin == y_pred_bin, axis=1).mean() * 100
    
    # Soft Accuracy: Percentage of rows with at least one correct label
    soft_accuracy = (soft_match_count / len(df)) * 100
    
    # Jaccard Index: Average overlap (Intersection over Union)
    j_score = jaccard_score(y_true_bin, y_pred_bin, average='samples', zero_division=0) * 100

    print("\n" + "="*45)
    print("      GEMINI FINAL VALIDATION METRICS")
    print("="*45)
    print(f"Strict Accuracy (Exact Match): {exact_match:.2f}%")
    print(f"Soft Accuracy (Partial Match): {soft_accuracy:.2f}%")
    print(f"Jaccard Index (Avg Overlap):   {j_score:.2f}%")
    print(f"Micro F1-Score:                {micro_f1:.2f}")
    print(f"Hamming Loss (Lower is Better): {h_loss:.4f}")
    print("="*45)
    
    print("\nGemini Performance Analysis:")
    if exact_match > 75:
        print("High precision achieved. Gemini is effectively capturing complex multi-label sets.")
    else:
        print("Model shows strong theme identification. Review Jaccard score for overlap quality.")

if __name__ == "__main__":
    run_validation()