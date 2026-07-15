import pandas as pd
import numpy as np
import os

def explore_language(name, path):
    print(f"=== {name.upper()} DATA EXPLORATION ===")
    df = pd.read_csv(os.path.join(path, "labels.csv"))
    
    # 1. Total turns
    total_turns = df['turn_id'].nunique()
    print(f"Total turns: {total_turns}")
    
    # 2. Total pauses
    total_pauses = len(df)
    print(f"Total pauses: {total_pauses}")
    
    # 3. Class balance
    balance = df['label'].value_counts()
    balance_pct = df['label'].value_counts(normalize=True) * 100
    print("Class balance (labels count):")
    for val, count in balance.items():
        print(f"  {val}: {count} ({balance_pct[val]:.1f}%)")
        
    # 4. Average pauses per turn
    pauses_per_turn = df.groupby('turn_id')['pause_index'].count()
    print(f"Pauses per turn: mean={pauses_per_turn.mean():.2f}, min={pauses_per_turn.min()}, max={pauses_per_turn.max()}")
    
    # 5. Pause duration distribution
    df['pause_duration'] = df['pause_end'] - df['pause_start']
    print("Pause duration by class:")
    for label in ['hold', 'eot']:
        sub = df[df['label'] == label]['pause_duration']
        print(f"  {label.upper()}: mean={sub.mean():.3f}s, std={sub.std():.3f}s, min={sub.min():.3f}s, max={sub.max():.3f}s")
        print(f"    percentiles (25, 50, 75, 90): {np.percentile(sub, [25, 50, 75, 90])}")
    print()

def main():
    base_dir = "/home/simran/speedrun/data/eot_handout/eot_data/"
    explore_language("English", os.path.join(base_dir, "english"))
    explore_language("Hindi", os.path.join(base_dir, "hindi"))

if __name__ == "__main__":
    main()
