import os
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import joblib
import json
from pathlib import Path

from data.make_dataset import prepare_player_sequences, dataloader_create
from models.models import RNN, Transformer, LogRNN, LogTransformer 
from sklearn.preprocessing import MaxAbsScaler



# --- Setting Config --- 

PROCCESSED_DATA_DIR = "data/processed_data.csv"
SALAH_DIR = "data/salah.csv"
ACTIVE_DIR = "data/active_players.csv"
SAVE_DIR = "saved_models_and_visualizations"
LOG_DIR =  "logs"
HYPERPARAM_CONFIG_PATH =  "configs/best_params.yaml"
SCALER_PATH = os.path.join(SAVE_DIR, "fitted_scaler.joblib")



BATCH_SIZE = 128
MAX_EPOCHS = 30 
PATIENCE = 7    
USE_GPU = torch.cuda.is_available()


def main():
    print(f"--- Training Script ---")


    # --- Load Hyperparameter Config ---
    with open(HYPERPARAM_CONFIG_PATH, 'r') as f:
        all_hyperparams_config = yaml.safe_load(f)


    # --- Load Data --- change salah dir
    sequences, masks, targets = prepare_player_sequences(PROCCESSED_DATA_DIR, target_col="total_points", max_gws_in_sequence= 38)
    
    sequences_log, masks_log, targets_log = prepare_player_sequences(PROCCESSED_DATA_DIR, target_col="log1p_total_points", max_gws_in_sequence= 38)


    n_features = sequences.shape[2]
    seq_len = sequences.shape[1]




    # --- Prepare Scaler ---
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
  
    else:
        scaler = MaxAbsScaler()
        train_idx_for_scaler = np.arange(0, int(sequences.shape[0] * 0.7))
        scaler.fit(sequences[train_idx_for_scaler].reshape(-1, n_features))
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)



    # --- Define Models to Train ---


    # (model_variant_name, base_model_type, use_log_dataloader)
    models_to_train_config = [
        ("RNN", "RNN", False), 
        ("Transformer", "Transformer", False), 
        ("LogRNN", "RNN", True),
        ("LogTransformer", "Transformer", True)
    ]

    for model_variant_name, base_model_type, use_log_dl in models_to_train_config:
        print(f"\n\n--- Preparing to Train: {model_variant_name} ---")

        
        # --- Create DataLoaders ---
        if use_log_dl:
            train_loader, val_loader, test_loader = dataloader_create(
                sequences_log, masks_log, targets_log, scaler= scaler, batch_size=BATCH_SIZE
            )
        else:
            train_loader, val_loader, test_loader = dataloader_create(
                sequences, masks, targets, scaler= scaler, batch_size=BATCH_SIZE
            )




        # --- Initializing Models ---

        if base_model_type == "RNN":
            model_hparams = all_hyperparams_config.get('rnn_best_params', {}).copy()
            model_hparams['n_features'] = n_features
        elif base_model_type == "Transformer":
            model_hparams = all_hyperparams_config.get('transformer_best_params', {}).copy()
            model_hparams["input_features_dim"] = n_features

 


        print(f"Initializing model {model_variant_name} with params: {model_hparams}")
        try:
            if model_variant_name == "RNN": model_instance = RNN(**model_hparams)
            elif model_variant_name == "Transformer": model_instance = Transformer(**model_hparams)
            elif model_variant_name == "LogRNN": model_instance = LogRNN(**model_hparams)
            elif model_variant_name == "LogTransformer": model_instance = LogTransformer(**model_hparams)
        except Exception as e:
            print(f"Error initializing model {model_variant_name} with hparams {model_hparams}: {e}") 



        # --- Callbacks and Logger ---
        csv_logger = CSVLogger(os.path.join(LOG_DIR, model_variant_name), name="training_logs")

        monitor_metric_es = "val_loss_log_transformed" if use_log_dl else "val_loss"

        # We want to have the model with best real loss not log of the loss
        monitor_metric_ckpt = "val_loss" 

        early_stop = EarlyStopping(monitor=monitor_metric_es, patience=PATIENCE, verbose=True, mode="min", min_delta=0.0001)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        
        # new implementation that wasn't in model selection to get the best model in training
        model_ckpt_dir = os.path.join(SAVE_DIR, model_variant_name)
        os.makedirs(model_ckpt_dir, exist_ok=True)

        checkpoint = ModelCheckpoint(
            dirpath=model_ckpt_dir,
            filename=f"{model_variant_name}-best-{{epoch}}-{{{monitor_metric_ckpt}:.4f}}",
            save_top_k=1, monitor=monitor_metric_ckpt, mode="min", verbose=True
        )


        # --- Starting Training ---
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            logger=csv_logger,
            callbacks=[early_stop, checkpoint, lr_monitor],
            gradient_clip_val=1.0,
            accelerator="gpu" if USE_GPU else "cpu",
            devices=1 if USE_GPU else None,
        )

        print(f"\n--- Starting Training for {model_variant_name} ---")
        trainer.fit(model_instance, train_dataloaders=train_loader, val_dataloaders=val_loader)

        print(f"\n--- Training Finished for {model_variant_name} ---")

        print(f"Best validation score ({monitor_metric_ckpt}): {checkpoint.best_model_score}")

        summary = {
            'model_variant_name': model_variant_name,
            'best_checkpoint_path': checkpoint.best_model_path,
            'best_val_score_monitored': checkpoint.best_model_score.item() if checkpoint.best_model_score else None,
            'final_epoch': trainer.current_epoch,
            'hyperparameters_used': model_hparams,
        }

        summary_path = os.path.join(model_ckpt_dir, f"{model_variant_name}_summary.json")

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent = 4,)






        # some fundamental testing before eval.py

        print(f"\n--- Testing best model ({model_variant_name}) on Test Set ---")

        test_results = trainer.test(model_instance, dataloaders=test_loader, ckpt_path='best', verbose=False)

        print(f"Test results for {model_variant_name}: {test_results[0]}")

        test_results_path = os.path.join(model_ckpt_dir, f"{model_variant_name}_test_results.json")

        serializable_test_results = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in test_results[0].items()}
        with open(test_results_path, 'w') as f:
            json.dump(serializable_test_results, f, indent=4)


    print(f"\n\n--- All Configured Models Trained ---")

if __name__ == "__main__":
    main()