import pandas as pd
import argparse
import sys

def check_leakage(file_path):
    print(f"Scanning {file_path} for potential leakage...")
    
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, nrows=100) # Scan headers only
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 1. Suspicious Keywords
    suspicious_keywords = [
        'diag', 'diagnosis', 'medication', 'medicine', 'prescription', 'rx',
        'treat', 'therapy', 'outcome', 'future', 'target', 'label',
    ]
    
    found_suspects = []
    for col in df.columns:
        col_lower = col.lower()
        for kw in suspicious_keywords:
            if kw in col_lower and col_lower != 'sick_onset': # sick_onset is target
                found_suspects.append((col, kw))
    
    if found_suspects:
        print("\n[WARNING] Potential Leakage Detected! Check these columns:")
        for col, kw in found_suspects:
            print(f" - {col} (matches keyword '{kw}')")
    else:
        print("\n[PASS] No obvious suspicious keywords found in column names.")

    # 2. Future Timestamp Check (if applicable)
    # This requires knowing the index time column, which isn't generic here.
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to CSV or Parquet file')
    args = parser.parse_args()
    
    check_leakage(args.file_path)
