import pandas as pd
import os

def prepare_data():
    input_file = "bodywash-train.xlsx"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    print("Aggregating training data...")
    # Load Excel
    df = pd.read_excel(input_file)
    df.columns = df.columns.str.strip()
    # Drop rows where the review is empty
    df = df.dropna(subset=['Core Item'])
    # This fixes the most common encoding artifacts
    df['Core Item'] = df['Core Item'].str.encode('ascii', 'ignore').str.decode('ascii')
    
    aggregated = df.groupby('Core Item')['Level 1 Factors'].apply(lambda x: str(list(set(x)))).reset_index()
    
    aggregated.to_csv("aggregated_train.csv", index=False)
    print(f"Success! Created aggregated_train.csv with {len(aggregated)} unique reviews.")

if __name__ == "__main__":
    prepare_data()