import pandas as pd
import json
import os
import gc

def process_local_dataset():
    csv_file = r"C:\Users\S A\Downloads\kcc_dataset.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}. Make sure the download has completely finished (it should not end in .crdownload)!")
        return

    print(f"Processing massive local CSV in chunks to save RAM: {csv_file}")
    
    # Define keywords for our 3 intents
    sell_keywords = ['sell', 'buyer', 'ammu', 'bechna', 'sale']
    price_keywords = ['price', 'rate', 'cost', 'bhav', 'dhara', 'market']
    demand_keywords = ['demand', 'need', 'require', 'kharidne']
    
    sell_queries, price_queries, demand_queries = [], [], []
    
    chunk_size = 100000
    try:
        # Assuming the column containing the farmer query is named 'QueryText' or 'query'
        # If the column name is different, we might need to adjust this. 
        # We will try to read without specifying columns first to be safe, taking the first text column.
        
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
            chunk = chunk.dropna()
            
            # Try to find the query column
            query_col = None
            for col in chunk.columns:
                if 'query' in str(col).lower() or 'text' in str(col).lower() or 'question' in str(col).lower():
                    query_col = col
                    break
                    
            if not query_col:
                query_col = chunk.columns[0] # Fallback to first column
                
            queries = chunk[query_col].astype(str).str.lower()
            
            if len(sell_queries) < 500:
                sell_queries.extend(queries[queries.str.contains('|'.join(sell_keywords))].tolist())
                
            if len(price_queries) < 500:
                price_queries.extend(queries[queries.str.contains('|'.join(price_keywords))].tolist())
                
            if len(demand_queries) < 500:
                demand_queries.extend(queries[queries.str.contains('|'.join(demand_keywords))].tolist())
                
            del chunk
            gc.collect()
            
            if len(sell_queries) >= 500 and len(price_queries) >= 500 and len(demand_queries) >= 500:
                print("Found enough training data! Stopping early to save time.")
                break
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Keep exactly 500
    sell_queries = sell_queries[:500]
    price_queries = price_queries[:500]
    demand_queries = demand_queries[:500]

    print(f"Extracted {len(sell_queries)} sell, {len(price_queries)} price, {len(demand_queries)} demand.")

    intents_data = {
        "intents": [
            {"tag": "sell_produce", "patterns": sell_queries},
            {"tag": "check_price", "patterns": price_queries},
            {"tag": "check_demand", "patterns": demand_queries}
        ]
    }

    output_file = 'intents.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(intents_data, f, indent=4)

    print(f"\n✅ SUCCESS! Replaced intents.json with real Kaggle data!")

if __name__ == "__main__":
    process_local_dataset()
