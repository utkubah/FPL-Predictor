import os
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data.make_dataset import prepare_player_sequences, dataloader_create, scale_3d_sequences
from models.models import RNN, Transformer, LogRNN, LogTransformer 
from sklearn.preprocessing import MaxAbsScaler 

# --- Setting Config ---
PROCCESSED_DATA_DIR = "data/processed_data.csv" 
ACTIVE_DIR = "data/active_players.csv"
SALAH_DIR = "data/salah.csv"           
SAVE_DIR = "saved_models_and_visualizations"
SCALER_PATH = os.path.join(SAVE_DIR, "fitted_scaler.joblib")


BATCH_SIZE = 256 #we can use higher batch size in eval
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu") # add later

def main_evaluate():
    print(f"--- Evaluation Script ---")

    # --- Load Scaler ---
    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded scaler from {SCALER_PATH}")

    # --- Prepare Test DataLoaders ---

    sequences_orig, masks_orig, targets_orig_full_dataset = prepare_player_sequences(
        PROCCESSED_DATA_DIR, target_col="total_points", max_gws_in_sequence=38)
    
    _, _, test_loader_orig = dataloader_create(
        sequences_orig, masks_orig, targets_orig_full_dataset, scaler=scaler, batch_size=BATCH_SIZE
    )

    sequences_log, masks_log, targets_log_transformed_full_dataset = prepare_player_sequences(
        PROCCESSED_DATA_DIR, target_col="log1p_total_points", max_gws_in_sequence=38
    )
    _, _, test_loader_log = dataloader_create(
        sequences_log, masks_log, targets_log_transformed_full_dataset, scaler=scaler, batch_size=BATCH_SIZE
    )

    # Get the original targets of for later comparison
    test_split_ratio = 0.15
    test_start_index = int(targets_orig_full_dataset.shape[0] * (1 - test_split_ratio))
    original_test_set_targets_np = targets_orig_full_dataset[test_start_index:]


    # --- Models to Evaluate ---
    models_to_evaluate_config = [
        ("RNN", False),
        ("Transformer", False),
        ("LogRNN", True),
        ("LogTransformer", True)
    ]

    all_evaluation_results = {}
    salah_predictions_all_models = {} 
    active_players_predictions_all_models = {}


    for model_variant_name, use_log_dl_for_this_model in models_to_evaluate_config:
        print(f"\n\n--- Evaluating Model: {model_variant_name} ---")
        
        # we get the best model checkpoint from summary
        model_ckpt_dir = os.path.join(SAVE_DIR, model_variant_name)
        summary_path = os.path.join(model_ckpt_dir, f"{model_variant_name}_summary.json")

        checkpoint_path_from_summary = None

        with open(summary_path, 'r') as f:
            train_summary = json.load(f)
        checkpoint_path_from_summary = train_summary.get('best_checkpoint_path')
        

        checkpoint_path = checkpoint_path_from_summary
        print(f"Found best checkpoint from summary: {checkpoint_path}")
        
        
        try:
            if model_variant_name == "RNN": ModelClass = RNN
            elif model_variant_name == "Transformer": ModelClass = Transformer
            elif model_variant_name == "LogRNN": ModelClass = LogRNN
            elif model_variant_name == "LogTransformer": ModelClass = LogTransformer
            else: continue
            
            loaded_model = ModelClass.load_from_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"Error loading model {model_variant_name} from {checkpoint_path}: {e}")
            continue

        current_test_loader = test_loader_log if use_log_dl_for_this_model else test_loader_orig

        # only setting a trainer for eval
        trainer_eval = pl.Trainer(logger=False, accelerator="gpu" if USE_GPU else "cpu", devices=1 if USE_GPU else None)

        test_results_metrics = trainer_eval.test(loaded_model, dataloaders=current_test_loader, verbose=False)
        print(f"Trainer test_results for {model_variant_name}: {test_results_metrics[0]}")


        # --- Plotting the Testing Results ---

        loaded_model.eval()
        all_preds_model_scale_list = []

        with torch.no_grad():
            for batch_data in current_test_loader:
                seq, msk, _ = batch_data # Don't need target from loader for preds because we already created it in the beginning
                seq, msk = seq, msk
                pred_model_output = loaded_model(seq, msk)
                all_preds_model_scale_list.append(pred_model_output.cpu())
        all_preds_model_scale_tensor = torch.cat(all_preds_model_scale_list)

        if use_log_dl_for_this_model:
            final_predictions_np = torch.round(torch.expm1(all_preds_model_scale_tensor)).numpy().squeeze()
            plot_title_suffix = "(Log Model - Original Scale)"
        else:
            final_predictions_np = torch.round(all_preds_model_scale_tensor).numpy().squeeze()
            plot_title_suffix = "(Original Scale Model)"


        
        #min_len_test = min(len(final_predictions_np), len(original_test_set_targets_np))
        #for plotting add this if necessary
        final_predictions_np_plot = final_predictions_np
        original_test_targets_np_plot = original_test_set_targets_np

        plot_dir = os.path.join(SAVE_DIR, f"{model_variant_name}_evaluation_plots")
        os.makedirs(plot_dir, exist_ok=True)



        # --- Residuals PLot ---
        residuals = original_test_targets_np_plot - final_predictions_np_plot
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=max(30, int(residuals.max() - residuals.min()) + 1 if residuals.size > 0 else 30))
        plt.xlabel("Residual (True - Predicted)"); plt.ylabel("Frequency")
        plt.title(f"{model_variant_name} - Test Set: Residual Plot {plot_title_suffix}")
        plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(plot_dir, f"{model_variant_name}_residuals.png")); plt.close()
        print(f"Evaluation plots saved in {plot_dir}")
        
        all_evaluation_results[model_variant_name] = {
            "trainer_metrics": test_results_metrics,
            "checkpoint_used": checkpoint_path
        }



        # --- Cumulative Accuracy Plot ---
        errors_abs_test = np.abs(original_test_targets_np_plot - final_predictions_np_plot)
        tolerances_plot = np.arange(0, int(errors_abs_test.max()) + 2 if errors_abs_test.size > 0 else 5) 
        accuracies_at_tolerance_plot = [np.mean(errors_abs_test <= tol) * 100 for tol in tolerances_plot]
        plt.figure(figsize=(10, 6))
        plt.plot(tolerances_plot, accuracies_at_tolerance_plot, marker='o', linestyle='-')
        
        plt.xlabel("Tolerance (+/- X points)"); plt.ylabel("% Predictions within Tolerance")
        plt.title(f"{model_variant_name} - Cumulative Accuracy {plot_title_suffix}"); plt.xticks(tolerances_plot)
        plt.yticks(np.arange(0, 101, 10))
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(SAVE_DIR, f"{model_variant_name}_test_cumulative_accuracy.png")); plt.close()


        # --- Salah Evaluation ---
        print(f"\n--- Preparing data for Salah for model {model_variant_name} ---")
        player_target_col = "log1p_total_points" if use_log_dl_for_this_model else "total_points"
        
        p_sequences, p_masks, p_targets_model_scale = prepare_player_sequences(
            SALAH_DIR, target_col=player_target_col, max_gws_in_sequence=38
        )
        # also get original un-transformed targets for Salah for plotting
        _, _, p_targets_original_scale = prepare_player_sequences(
            SALAH_DIR, target_col="total_points", max_gws_in_sequence=38
        )
        
        #scaling and predicting
        p_sequences_scaled = scale_3d_sequences(p_sequences, scaler)
        p_sequences_tensor = torch.tensor(p_sequences_scaled, dtype=torch.float32)
        p_masks_tensor = torch.tensor(p_masks, dtype=torch.bool)
            
        with torch.no_grad():
            p_preds_model_scale = loaded_model(p_sequences_tensor, p_masks_tensor)
            
        if use_log_dl_for_this_model:
            p_preds_final_np = torch.round(torch.expm1(p_preds_model_scale.cpu())).numpy().squeeze()
        else:
            p_preds_final_np = torch.round(p_preds_model_scale.cpu()).numpy().squeeze()
            
        salah_predictions_all_models[model_variant_name] = {
            "predictions": p_preds_final_np.tolist(), # Store as list for JSON
            "true_original": p_targets_original_scale.squeeze().tolist() # Original points of Salah
        }
        
        active_mae = np.mean(np.abs(p_preds_final_np - p_targets_original_scale))
        all_evaluation_results[model_variant_name][f"MAE_Salah"] = float(active_mae)

        print(f"MAE for Salah with {model_variant_name}: {active_mae:.4f}")
   

        
        # --- Active Players Evaluation ---
        print(f"\n--- Preparing data for Active Players for model {model_variant_name} ---")
        active_target_col = "log1p_total_points" if use_log_dl_for_this_model else "total_points"
        active_seq, active_masks, _ = prepare_player_sequences( # We don't need targets_model_input for preds
            ACTIVE_DIR, target_col=active_target_col, max_gws_in_sequence=38
        )

        # Get original un-trasnformed targets for active players for comparison
        _, _, active_targets_original_truth = prepare_player_sequences(
            ACTIVE_DIR, target_col="total_points", max_gws_in_sequence=38
        )



        active_seq_scaled = scale_3d_sequences(active_seq, scaler)
        active_seq_tensor = torch.tensor(active_seq_scaled, dtype=torch.float32)
        active_masks_tensor = torch.tensor(active_masks, dtype=torch.bool)

        with torch.no_grad():
            active_preds_model_out = loaded_model(active_seq_tensor, active_masks_tensor)

        if use_log_dl_for_this_model:
            active_preds_final = torch.round(torch.expm1(active_preds_model_out.cpu())).numpy().squeeze()
        else:
            active_preds_final = torch.round(active_preds_model_out.cpu()).numpy().squeeze()
            
        active_players_predictions_all_models[model_variant_name] = {
            "predictions": active_preds_final.tolist(), 
            "true_original": active_targets_original_truth.tolist() 
        }
        
        # Calculate MAE for active players for this model
        active_mae = np.mean(np.abs(active_preds_final - active_targets_original_truth))
        all_evaluation_results[model_variant_name][f"MAE_ActivePlayers"] = float(active_mae)
        print(f"MAE for Active Players with {model_variant_name}: {active_mae:.4f}")



    


    # --- Save all evaluation results (including trainer metrics) to a single file ---
    overall_eval_summary_path = os.path.join(SAVE_DIR, "all_models_evaluation_summary.json")
    with open(overall_eval_summary_path, 'w') as f:
        json.dump(all_evaluation_results, f, indent=4)
    print(f"\n\n--- Overall Evaluation Summary (trainer.test metrics) saved to: {overall_eval_summary_path} ---")


    # --- Plot Specific Player (Salah) Comparison Across All Models ---
    if salah_predictions_all_models:
        plt.figure(figsize=(15, 8))
        
        # Get Salah's true original points once (assuming it's the same for all models)
        first_model_name = list(salah_predictions_all_models.keys())[0]
        salah_true_original_points_np = np.array(salah_predictions_all_models[first_model_name]["true_original"])
        
        plt.plot(salah_true_original_points_np, label=f'Salah - True Points (Original)', marker='o', linewidth=2.5, markersize=5, color='black')

        for model_name_plotted, data in salah_predictions_all_models.items():
            plt.plot(np.array(data["predictions"]), label=f'Salah - {model_name_plotted} Preds', marker='x', linestyle='--', markersize=4, alpha=0.8)
        
        plt.xlabel(f"Gameweek Instance for Salah")
        plt.ylabel("Total Points")
        plt.title(f"Comparison of Model Predictions for Salah")
        plt.legend(loc='upper left', bbox_to_anchor=(1,1)) 
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        salah_plot_path = os.path.join(SAVE_DIR, f"salah_all_models_comparison.png")
        plt.savefig(salah_plot_path)
        print(f"\nSalah comparison plot saved to: {salah_plot_path}")
        plt.close()



if __name__ == "__main__":
    main_evaluate()