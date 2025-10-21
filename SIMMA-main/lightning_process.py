import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from models.LSTD.MultiModalLinear import MultiModalLinear
from models.CMTA.MultiModalsFormer import MultiModalFormer
import time
import csv


class MILRegressionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        lstd_ModelClass = globals()[args.LSTD]
        self.lstd_model = lstd_ModelClass(args=args)

        self.cmta_modelClass = globals()[args.CMTA]
        self.cmta_model = self.cmta_modelClass(args=args)

        self.mse = nn.MSELoss()
        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        self.validation_step_outputs = []

        self.epoch_start_time = None
        self.stats_csv_path = "epoch_stats.csv"

    def forward(self, x):
        if self.args.use_linear:
            x = self.lstd_model(x.float())
        if self.args.use_align or self.args.use_transformer:
            x = self.cmta_model(x.float())

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y.view(-1, 1))
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_epoch",
                "interval": "epoch",  # Explicitly set to 'epoch'
                "frequency": 1,
            },
        }

    def validation_step(self, batch, batch_idx):
        self.evaluation_step(batch)

    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch)

    def on_validation_epoch_end(self):
        self.on_evaluation_epoch_end(stage='val')

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end(stage='test')

    def evaluation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.reshape((-1, 1)))

        output = {'loss': loss.detach(), 'preds': y_hat.detach(), 'labels': y.detach()}
        self.validation_step_outputs.append(output)

    def on_evaluation_epoch_end(self, stage='val'):
        if not self.validation_step_outputs:
            return

        losses = torch.stack([x['loss'] for x in self.validation_step_outputs])
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        labels = torch.cat([x['labels'] for x in self.validation_step_outputs])

        preds = preds.squeeze()

        avg_loss = losses.mean()
        avg_mae = self.mean_absolute_error(preds, labels)
        avg_rmse = torch.sqrt(self.mean_squared_error(preds, labels))

        self.log_dict(
            {f'{stage}_loss_epoch': avg_loss, f'avg_{stage}_mae': avg_mae, f'avg_{stage}_rmse': avg_rmse},
            on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if self.logger and self.logger.log_dir:
            preds_list = preds.cpu().numpy().flatten().tolist()
            labels_list = labels.cpu().numpy().flatten().tolist()
            df_out = pd.DataFrame({'predictions': preds_list, 'labels': labels_list})
            os.makedirs(self.logger.log_dir, exist_ok=True)
            df_out.to_csv(os.path.join(self.logger.log_dir, f'{stage}_predictions_epoch_{self.current_epoch}.csv'),
                          index=False)

        self.validation_step_outputs.clear()
