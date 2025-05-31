import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



# --- PyTorch Dataset Class ---
class FPLPlayerSequenceDataset(Dataset): 
    """
    - sequence_of_features: A tensor of players and their features across gameweeks
    - target_points: A scalar tensor representing the points in the gameweek
    - attention_masks: A scalar tensor representing which gameweek information is available at the time
    """
    def __init__(self, sequences, targets, attention_masks):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.attention_masks = torch.tensor(attention_masks, dtype=torch.bool) 
        # Attention mask indicates which parts of the sequence are real data vs. padding

        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)



    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.attention_masks[idx], self.targets[idx]
    
        



# --- Core Data Processing and Sequencing Function ---

# function gets the processed data and the target column and spits out
# 3 variables: 3D tensor of sequences, an attention mask to prevent looking into "future" and the targets
# we will end up with data that can be used in FPLPlayerSequenceDataset and moreover in Dataloader
def prepare_player_sequences(df_path: str, 
                             target_col: str, 
                             max_gws_in_sequence: int):


    print(f"--- Starting Data Preparation for Sequences from {df_path} ---")

    # --- Load Data ---
    try:
        df = pd.read_csv(df_path)
        print(f"Successfully loaded data. Shape: {df.shape}")
    except Exception as e:
        print(f"ERROR: Could not load data. {e}")
        return None, None, None

    # --- Data Cleaning and Initial Type Conversion ---

    print("\n--- Cleaning Data and Selecting Numeric Features ---")
    

    # We aim to keep only numerical features
    potential_cols_to_drop = []
    selected_feature_names = []

    # here we drop the ones that are already lagged and therefore unnecessary and can cause data leakage
    lagged_features = ["assists", "bonus", "bps", "clean_sheets", "creativity", "goals_conceded", 
                           "goals_scored", "ict_index", "influence", "minutes", "own_goals", "penalties_missed", 
                           "penalties_saved", "red_cards", "saves", "starts", "threat", "yellow_cards", 
                           "xP", "expected_assists", "expected_goal_involvements", "expected_goals", "expected_goals_conceded", 
                           "value", "selected", "transfers_balance", "transfers_in", "transfers_out", "team_a_score", "team_h_score"]  


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







    # --- Create Sequences ---
    print(f"\n--- Creating Fixed-Length Sequences (Max Gameweeks: {max_gws_in_sequence}) ---")
    all_sequences_list  = []
    all_targets_list  = []
    all_masks_list = []


    grouped_by_player = df_subset.groupby("player_id")

    # dropping the total_point columns 
    to_drop = {"total_points", "log1p_total_points"}
    selected_feature_names = [col for col in selected_feature_names if col.strip() not in to_drop]

    for player_id, player_data in grouped_by_player:

        # For each player, we can generate multiple sequences:
        # Seq1: history up to GW1 (i.e., only GW1 data used), predict GW1
        # Seq2: history up to GW2 (i.e., GW1, GW2 data used), predict GW2
        # ...
        # SeqN: history up to GW(k), predict GW(k)

        #this could have caused a problem with data leakage however we took care of it
        #also this nature of the data explain why later GWs are easier to predict
         
        
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


# after getting the sequences we can use this function to create dataloaders
def dataloader_create(sequences, masks, targets, scaler, batch_size = 128):

    # different from the model_selection version we use a fitted scaler
    # train/val/test split
    # we do it a chronological way 
    n = sequences.shape[0]

    train_end = int(n * 0.70)          
    val_end   = train_end + int(n * 0.15)  

    train_idx = np.arange(0, train_end)
    val_idx   = np.arange(train_end, val_end)
    test_idx  = np.arange(val_end, n)


    sequences_train_scaled = scale_3d_sequences(sequences[train_idx], scaler)
    sequences_val_scaled = scale_3d_sequences(sequences[val_idx], scaler)
    sequences_test_scaled = scale_3d_sequences(sequences[test_idx], scaler)

    masks_train = masks[train_idx]
    targets_train = targets[train_idx]

    masks_val = masks[val_idx]
    targets_val = targets[val_idx]

    masks_test = masks[test_idx]
    targets_test = targets[test_idx]



    train_ds = FPLPlayerSequenceDataset(sequences_train_scaled, targets_train, masks_train) 
    val_ds = FPLPlayerSequenceDataset(sequences_val_scaled, targets_val, masks_val)     
    test_ds = FPLPlayerSequenceDataset(sequences_test_scaled, targets_test, masks_test)     




    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) 
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    return train_loader, val_loader, test_loader

#scaling is an important step as our data and especially our transformer model benefits from it
def scale_3d_sequences(sequence_array_3d, fitted_scaler):
    if sequence_array_3d.shape[0] == 0: # Handle empty array
        return sequence_array_3d 
    
    original_shape = sequence_array_3d.shape
    num_features_in_array = original_shape[2]
    
    # Reshape to 2D
    reshaped_array = sequence_array_3d.reshape(-1, num_features_in_array)
    
    # Transform using the FITTED scaler
    scaled_reshaped_array = fitted_scaler.transform(reshaped_array)
    
    # Reshape back to original 3D shape
    return scaled_reshaped_array.reshape(original_shape)