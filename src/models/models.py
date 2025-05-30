
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import math
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiStepLR


# custom weighted loss function

class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, underprediction_penalty=2.0):
        super().__init__()
        self.beta = beta
        self.underprediction_penalty = underprediction_penalty
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none', beta=beta)

    def forward(self, input_preds, target_actuals):
        loss = self.smooth_l1_loss(input_preds, target_actuals)

        penalty_weights = torch.ones_like(loss)
        penalty_weights[input_preds < target_actuals] = self.underprediction_penalty

        weighted_loss = loss * penalty_weights
        return weighted_loss.mean() 
    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.2, sequence_length = 38): 
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(sequence_length, 1, d_model)
        
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe.squeeze(1)
        return self.dropout(x)



class RNN(pl.LightningModule):
    def __init__(self, n_features, hidden_dim=128, n_layers=2, dropout=0.4, lr=1e-4, weight_decay = 1e-5, underprediction_penalty= None):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

        
        self.criterion = nn.SmoothL1Loss(beta=1.0)

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, mask):
        out, _ = self.lstm(x)
        lengths = mask.sum(dim=1)       
        idx = lengths - 1                
      
        
        last = out[torch.arange(out.size(0)), idx]
        return self.fc(last)
    

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        preds = self(x, mask)

        loss = self.criterion(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        preds = self(x, mask)

        #for validation step we want to be accurate to original points which are integers so we round the prediction

        preds_rounded = torch.round(preds) 


        loss = self.criterion(preds_rounded, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Accuracy within a tolerance (e.g., prediction is within +/- 1 point of true)
        tolerance = 1.0
        accuracy_tolerant = (torch.abs(preds_rounded - y) <= tolerance).float().mean()
        self.log("val_accuracy_tolerant_1pt", accuracy_tolerant, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        

        return loss

    def test_step(self, batch, batch_idx):
        x, mask, y = batch
        preds = self(x, mask)

        preds_rounded = torch.round(preds) 


        loss = self.criterion(preds_rounded, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        accuracy_exact = (preds_rounded == y).float().mean()
        self.log("test_accuracy_exact", accuracy_exact, prog_bar=False, on_step=False, on_epoch=True)

        
        # Accuracy within a tolerance (e.g., prediction is within +/- 1 point of true)
        tolerance = 1.0
        accuracy_tolerant = (torch.abs(preds_rounded - y) <= tolerance).float().mean()
        self.log("test_accuracy_tolerant_1pt", accuracy_tolerant, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",      # Reduce LR when the monitored quantity has stopped decreasing
                factor=0.1,      # Factor by which the learning rate will be reduced (new_lr = lr * factor)
                patience=3,      # Number of epochs with no improvement after which LR will be reduced
                verbose=True,    # Print a message when LR is reduced
                min_lr=1e-7      # Lower bound on the learning rate
            )
            
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss", # Metric to monitor for LR reduction
                    "interval": "epoch",   # Check scheduler at the end of each epoch
                    "frequency": 1         # Check every 1 epoch
                }
            }

    

class Transformer(pl.LightningModule):
    def __init__(self,
                 input_features_dim,
                 d_model = 256,
                 nhead = 4,
                 num_encoder_layers = 2,
                 dim_feedforward = 512,
                 dropout = 0.4,
                 sequence_length = 38,
                 lr = 5e-5,
                 weight_decay = 5e-4,
                 warmup_epochs = 5, #useful for transformers
                 milestones_after_warmup = [3,7],
                 multistep_gamma = 0.2,
                 underprediction_penalty= None
                 ):
        
        super().__init__()
        self.save_hyperparameters()


        # input projection layer necessary for positional encoding 
        self.input_projection = nn.Linear(input_features_dim, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        nn.init.zeros_(self.input_projection.bias)

        self.input_norm = nn.LayerNorm(d_model) #this solved it

        self.pos_encoder = PositionalEncoding(d_model, dropout, sequence_length=sequence_length)


        # encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)


        # prediction head that predicts a single value (total_points)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) 
        )

        self.criterion = nn.SmoothL1Loss(beta=1.0)

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.milestones_after_warmup = milestones_after_warmup
        self.multistep_gamma = multistep_gamma




    def forward(self, x, mask):
        x = self.input_projection(x) # [batch_size, seq_len, d_model] 
        x = self.input_norm(x)  
        x = nn.ReLU()(x) # Apply activation

        # Add positional encoding
        x = self.pos_encoder(x)
        

        # Pass through Transformer Encoder
        transformer_padding_mask = ~mask # Invert: True for padded, False for valid
        encoded_sequence = self.transformer_encoder(x, src_key_padding_mask=transformer_padding_mask) 

                
        # can't feed a 3d tensor into a linear so 
        batch_size = encoded_sequence.size(0)
        sequence_lengths = mask.sum(dim=1) 
        idx = sequence_lengths-1 # index of last valid gameweek

        # turn into 2D and use the final week that captures the context of previous weeks to decide the total points
        final_representation = encoded_sequence[torch.arange(batch_size), idx, :] 
        
        prediction = self.output_head(final_representation) # [batch_size, 1]
        return prediction


    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        preds = self(x, mask)

        loss = self.criterion(preds, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        preds = self(x, mask)

        preds_rounded = torch.round(preds) 


        loss = self.criterion(preds_rounded, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Accuracy within a tolerance (e.g., prediction is within +/- 1 point of true)
        tolerance = 1.0
        accuracy_tolerant = (torch.abs(preds_rounded - y) <= tolerance).float().mean()
        self.log("val_accuracy_tolerant_1pt", accuracy_tolerant, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss


    def test_step(self, batch, batch_idx):
        x, mask, y = batch
        preds = self(x, mask)

        preds_rounded = torch.round(preds) 


        loss = self.criterion(preds_rounded, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        accuracy_exact = (preds_rounded == y).float().mean()
        self.log("test_accuracy_exact", accuracy_exact, prog_bar=False, on_step=False, on_epoch=True)

        # Accuracy within a tolerance (e.g., prediction is within +/- 1 point of true)
        tolerance = 1.0
        accuracy_tolerant = (torch.abs(preds_rounded - y) <= tolerance).float().mean()
        self.log("test_accuracy_tolerant_1pt", accuracy_tolerant, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss




    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)



        # warm up for the transformer
        scheduler1 = LinearLR(optimizer, 
                              start_factor=0.01, 
                              end_factor=1.0, 
                              total_iters=self.warmup_epochs) 

      
            
    
        scheduler2 = MultiStepLR(optimizer,
                             milestones=self.milestones_after_warmup,
                             gamma=self.multistep_gamma) # Factor to reduce LR by (e.g., 0.1 or 0.2)




        final_scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[self.warmup_epochs])


        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": final_scheduler,
                "interval": "epoch",  
                "frequency": 1

            }
        }



class LogRNN(RNN): # Inherits from RNN
    def __init__(self, n_features, hidden_dim=64, n_layers=3, dropout=0.4, lr=1e-4, weight_decay=1e-5,underprediction_penalty=1.0):
        super().__init__(n_features)


        self.underprediction_penalty = underprediction_penalty
        self.criterion = WeightedSmoothL1Loss(underprediction_penalty=self.underprediction_penalty) #when 1.0 same as SmoothL1

    def validation_step(self, batch, batch_idx):
        x, mask, y_log = batch # y_log is log(1+y_true)
        preds_log = self(x, mask) # Model predicts log(1+y_pred)

        loss = self.criterion(preds_log, y_log) # Loss is on log-transformed values
        self.log("val_loss_log_transformed", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Inverse transform for interpretable metrics
        preds_orig_scale = torch.expm1(preds_log) # expm1 is exp(x) - 1
        y_orig_scale = torch.expm1(y_log)
        preds_rounded = torch.round(preds_orig_scale)

        real_loss = WeightedSmoothL1Loss()(preds_rounded, y_orig_scale) 
        self.log("val_loss", real_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        tolerance = 1.0
        accuracy_tolerant = (torch.abs(preds_rounded - y_orig_scale) <= tolerance).float().mean()
        self.log("val_accuracy_tolerant_1pt", accuracy_tolerant, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss 

    def test_step(self, batch, batch_idx):
        x, mask, y_log = batch
        preds_log = self(x, mask)
        loss = self.criterion(preds_log, y_log)
        self.log("test_loss_log_transformed", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        preds_orig_scale = torch.expm1(preds_log)
        y_orig_scale = torch.expm1(y_log)
        preds_rounded = torch.round(preds_orig_scale)

        real_loss = WeightedSmoothL1Loss()(preds_rounded, y_orig_scale)
        self.log("test_loss", real_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        accuracy_exact = (preds_rounded == y_orig_scale).float().mean()
        self.log("test_accuracy_exact", accuracy_exact, prog_bar=False, on_step=False, on_epoch=True)

        tolerance = 1.0
        accuracy_tolerant = (torch.abs(preds_rounded - y_orig_scale) <= tolerance).float().mean()
        self.log("test_accuracy_tolerant_1pt", accuracy_tolerant, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss




class LogTransformer(Transformer): # Inherits from your existing Transformer class
    def __init__(self, input_features_dim, d_model=512, nhead=4, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.4, sequence_length=38, lr=5e-5,
                 weight_decay=5e-4, warmup_epochs=5, milestones_after_warmup=[3,7],
                 multistep_gamma=0.2, underprediction_penalty=2.0):
        super().__init__(input_features_dim)

        self.underprediction_penalty = underprediction_penalty
        self.criterion = WeightedSmoothL1Loss(underprediction_penalty=self.underprediction_penalty) 

    def validation_step(self, batch, batch_idx):
        x, mask, y_log = batch
        preds_log = self(x, mask)
        loss = self.criterion(preds_log, y_log)
        self.log("val_loss_log_transformed", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        preds_orig_scale = torch.expm1(preds_log)
        y_orig_scale = torch.expm1(y_log)
        preds_rounded = torch.round(preds_orig_scale)

        real_loss = WeightedSmoothL1Loss(beta=0.1)(preds_rounded, y_orig_scale)
        self.log("val_loss", real_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        tolerance = 1.0
        accuracy_tolerant = (torch.abs(preds_rounded - y_orig_scale) <= tolerance).float().mean()
        self.log("val_accuracy_tolerant_1pt", accuracy_tolerant, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, mask, y_log = batch
        preds_log = self(x, mask)
        loss = self.criterion(preds_log, y_log)
        self.log("test_loss_log_transformed", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        preds_orig_scale = torch.expm1(preds_log)
        y_orig_scale = torch.expm1(y_log)
        preds_rounded = torch.round(preds_orig_scale)

        mae_orig_scale = WeightedSmoothL1Loss(beta=0.1)(preds_rounded, y_orig_scale)
        self.log("test_loss", mae_orig_scale, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        accuracy_exact = (preds_rounded == y_orig_scale).float().mean()
        self.log("test_accuracy_exact", accuracy_exact, prog_bar=False, on_step=False, on_epoch=True)

        tolerance = 1.0
        accuracy_tolerant = (torch.abs(preds_rounded - y_orig_scale) <= tolerance).float().mean()
        self.log("test_accuracy_tolerant_1pt", accuracy_tolerant, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
