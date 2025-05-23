import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



# --- PyTorch Dataset Class ---
class FPLPlayerSequenceDataset(Dataset): # add pytorch
    """
    - sequence_of_features: A tensor of players and their features across gameweeks
    - target_points: A scalar tensor representing the points in the gameweek
    - attention_masks: A scalar tensor representing which gameweek information is available at the time
    """
    def __init__(self, sequences, targets, attention_masks):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

        # Attention mask indicates which parts of the sequence are real data vs. padding
        self.attention_masks = torch.tensor(attention_masks, dtype=torch.bool) # or float32 if model expects that

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.attention_masks[idx], self.targets[idx]
    




# --- Core Data Processing and Sequencing Function ---
def prepare_player_sequences(df_path: str, 
                             target_col: str, 
                             max_gws_in_sequence: int):


    print(f"--- Starting Data Preparation for Sequences from {df_path} ---")

    # 1. Load Data
    try:
        df = pd.read_csv(df_path)
        print(f"Successfully loaded data. Shape: {df.shape}")
    except Exception as e:
        print(f"ERROR: Could not load data. {e}")
        return None, None, None

    # 2. Data Cleaning and Initial Type Conversion

    print("\n--- Cleaning Data and Selecting Numeric Features ---")
    

    # We aim to keep only numerical features
    potential_cols_to_drop = []
    selected_feature_names = []

    # here we drop the ones that are already lagged and therefore unnecessary and can cause data leakage
    lagged_features = ["assists", "bonus", "bps", "clean_sheets", "creativity", "goals_conceded", 
                           "goals_scored", "ict_index", "influence", "minutes", "own_goals", "penalties_missed", 
                           "penalties_saved", "red_cards", "saves", "starts", "threat", "yellow_cards", 
                           "xP", "expected_assists", "expected_goal_involvements", "expected_goals", "expected_goals_conceded", 
                           "value", "selected", "transfers_balance", "transfers_in", "transfers_out"]

    #change this such that only in gw(k) they are dropped

    for col in df.columns:

        if col in lagged_features:
            continue

        # Attempt to convert columns to numeric, coercing errors to NaN
        # This helps identify columns that are non-numeric or mixed-type.
        try:
            df[col] = pd.to_numeric(df[col])
            if df[col].isnull().sum() > 0:  
                 # If conversion introduced NaNs, it means it had non-numeric strings.
                 print(f"Warning: Column '{col}' converted to numeric, but has NaNs after conversion. Consider imputation if keeping.")
            selected_feature_names.append(col)
        except ValueError:
            # an object column we can't easily use directly.
            if df[col].dtype == "object":
                potential_cols_to_drop.append(col)
                print(f"Column '{col}' is of type object and couldn't be converted to numeric. Marked for dropping.")


    
    
    

    if "player_id" in selected_feature_names:
        print(f"Note: '{"player_id"}' is in selected features. It will be part of the dense sequence.")

        #think about it


    print(f"Column '{lagged_features}' are dropped for preventing data leakage.")
    print(f"Columns dropped for other reasons: {potential_cols_to_drop}")
    
    print(f"Selected {len(selected_feature_names)} features for sequences: {selected_feature_names[:5]}...") # Show a sample

    
    # keep only necessary columns for sequencing + target    
    df_subset = df[selected_feature_names].copy()


    # filling NaN
    for col in selected_feature_names:
        if df_subset[col].isnull().any():
            print(f"Filling NaNs in feature column '{col}' with 0.")
            df_subset[col] = df_subset[col].fillna(0)
    
    # sorting data sequences must be  chronologically ordered
    print(f"\nSorting data by '{"player_id"}' and then '{"GW"}'...")
    df_subset = df_subset.sort_values(by=["player_id", "GW"])
    df_subset = df_subset.reset_index(drop=True)







    # 3. Create Sequences
    print(f"\n--- Creating Fixed-Length Sequences (Max Gameweeks: {max_gws_in_sequence}) ---")
    all_sequences_list  = []
    all_targets_list  = []
    all_masks_list = []


    grouped_by_player = df_subset.groupby("player_id")


    for player_id, player_data in grouped_by_player:

        # For each player, we can generate multiple sequences:
        # Seq1: history up to GW1 (i.e., only GW1 data used), predict GW1
        # Seq2: history up to GW2 (i.e., GW1, GW2 data used), predict GW2
        # ...
        # SeqN: history up to GW(k), predict GW(k)

        #this could have caused a problem with data leakage however we took care of it
         
        
        max_player_gw = player_data["GW"].max()


        for t in range(1, int(max_player_gw)+1): 
            
            # Iterate through each GW the player has data for, to use as end of history
            # History includes data up to and including gameweek "t". With information unavailable 
            # before the match start at t is dropped but information was preserved with lagged with previous matches  
            
            target_gw = t
            target_instance = player_data[player_data["GW"] == target_gw]

            if target_instance.empty: # safety
                continue

            current_target = target_instance[target_col].iloc[0]
            
 
             
            # dropping the target col 
            selected_feature_names = [
            col for col in selected_feature_names
            if col.strip() != target_col
            ]

            sequence_features = np.zeros((max_gws_in_sequence, len(selected_feature_names)), dtype=np.float32)


            # Mask is 1 for real data, 0 for padding. (Or True/False)
            attention_mask = np.zeros(max_gws_in_sequence, dtype=bool) 

            # Populate the sequence with data available up to gameweek `t`
            history_data_for_sample = player_data[player_data["GW"] <= t].copy()
            #do not forget the get rif of target col before creating sequences            
            for _, row in history_data_for_sample.iterrows():
                gw_idx = int(row["GW"]) -1 # Convert GW number (1-38) to 0-indexed slot
                if 0 <= gw_idx < max_gws_in_sequence:
                    sequence_features[gw_idx, :] = row[selected_feature_names].values.astype(np.float32)
                    
                    attention_mask[gw_idx] = True # Mark this timestep as real data
                
            all_sequences_list.append(sequence_features)
            all_targets_list.append(current_target)
            all_masks_list.append(attention_mask)

    if not all_sequences_list:
        print("ERROR: No sequences were created. Check data and logic.")
        return None, None, None, None

    sequences_np = np.array(all_sequences_list)
    targets_np = np.array(all_targets_list)
    masks_np = np.array(all_masks_list)

    print(f"Successfully created sequences. Shape: {sequences_np.shape}")
    print(f"Targets shape: {targets_np.shape}")
    print(f"Attention masks shape: {masks_np.shape}")
        
    return sequences_np, masks_np, targets_np

 
 
 # add standard scaler
 
""" 
# --- Main Execution ---
if __name__ == "__main__":

    sequences, attention_masks, targets, feature_names = prepare_player_sequences(
        df_path="C:/Users/asus/OneDrive/Desktop/projects/AI_project/data/processed_data.csv",
        target_col="total_points",
        max_gws_in_sequence=38
    )

    if sequences is not None:
        print(f"\nNumber of features per timestep: {sequences.shape[2]}")
        print(f"Feature names: {feature_names[:5]}...")

        print("\n--- Creating PyTorch Dataset and DataLoader ---")
        full_dataset = FPLPlayerSequenceDataset(sequences, targets, attention_masks)
        full_dataloader = DataLoader(full_dataset, batch_size=32, shuffle=False)

        print(f"PyTorch Dataset created with {len(full_dataset)} samples.")
        try:
            sample_seq, sample_mask, sample_target = next(iter(full_dataloader))
            print(f"Sample batch - Sequences shape: {sample_seq.shape}")      # (batch_size, MAX_GAMEWEEKS, num_features)
            print(f"Sample batch - Attention Mask shape: {sample_mask.shape}") # (batch_size, MAX_GAMEWEEKS)
            print(f"Sample batch - Targets shape: {sample_target.shape}")       # (batch_size, 1)
                
            # Check a mask and corresponding sequence part
            print(f"\nExample mask for first sample in batch:\n{sample_mask[0].numpy().astype(int)}")
            print(f"Corresponding sum of features for first sample (should be 0 where mask is 0 if padding is 0):\n{sample_seq[0].sum(axis=1).numpy()}")

        except StopIteration:
            print("Could not retrieve a sample batch from DataLoader.")
            
        print("\n--- Data Preparation Script Finished ---")
"""