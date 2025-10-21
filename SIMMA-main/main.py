import sys
import pandas as pd
import torch
import shutil
import lightning as pl
from lightning import seed_everything
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.loggers import CSVLogger
from dataset import MILDataset, MILRegressionDataModule
from lightning_process import MILRegressionModel
import os
import argparse

if __name__ == '__main__':
    train_dataset = 'AVEC2014'
    test_dataset = 'AVEC2014'

    parser = argparse.ArgumentParser(description='Personal information')

    # todo: add your data_dir path
    parser.add_argument('--data_dir', default='', type=str, help='data_dir')

    # todo: add your label_file path
    parser.add_argument('--label_file', default='', type=str,
                        help='data_dir')
    parser.add_argument('--train_data', default=[f'{train_dataset}-train'], nargs='+', help='train data')
    parser.add_argument('--val_data', default=[f'{test_dataset}-dev'], nargs='+', help='val data')
    parser.add_argument('--test_data', default=[f'{test_dataset}-test'], nargs='+', help='test data')

    parser.add_argument('--LSTD', default='MultiModalLinear', type=str, help='keys frames detect model')
    parser.add_argument('--CMTA', default='MultiModalFormer', type=str, help='modals fusion model')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
    parser.add_argument('--max_epochs', default=50, type=int, help='max_epochs')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning_rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
    parser.add_argument('--frame_interval', default=1, type=int, help='frame_interval')
    parser.add_argument('--join', default=1, type=int, help='join')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--seed', type=float, default=1, help='seed')
    parser.add_argument('--devices', type=int, default=[0], nargs='+', help='devices')

    parser.add_argument('--deep', default=True, type=bool, help='deep')
    parser.add_argument('--deep_type', default="iresnet50_base_1", type=str, help='deep_type')

    parser.add_argument('--audio', default=True, type=bool, help='audio')
    parser.add_argument('--audio_type', default='audio_pann_64_3',
                        type=str, help='audio_type')

    parser.add_argument('--rppg', default=False, type=bool, help='rppg')
    parser.add_argument('--rppg_type', default='HRV_3000_normal' if train_dataset == 'AVEC2013' else 'HRV_1000_normal',
                        type=str, help='rppg_type')

    parser.add_argument('--au', default=False, type=bool, help='openface')
    parser.add_argument('--openface_type', default='openface_au', type=str, help='openface_type')

    parser.add_argument('--use_linear', default=True, action='store_true', help='if use the LSTD')
    parser.add_argument('--use_align', default=True, action='store_true', help='if use the CMTA')
    parser.add_argument('--use_transformer', default=True, action='store_true', help='if use the Transformer')

    # Key frames Detect model
    parser.add_argument('--in_len', type=int, default=2000, help='input MTS length (T)')
    parser.add_argument('--kernel_size', type=int, default=45, help='move step of avg_pool')
    parser.add_argument('--single_modal_dims', type=int, default=128, help='dimension of each modal')
    parser.add_argument('--detector_dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--seg_len', type=list, default=None)

    # Modals fusion model
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')

    # Adaptive parameters
    parser.add_argument('--input_dims', default=0, type=int, help='attention_dims')
    parser.add_argument('--num_modals', default=0, type=int, help='total number of modals')
    parser.add_argument('--video_dims', default=0, type=int, help='deep_dims')
    parser.add_argument('--audio_dims', default=0, type=int, help='audio_dims')
    parser.add_argument('--au_dims', default=0, type=int, help='audio_dims')
    parser.add_argument('--rppg_dims', default=0, type=int, help='rppg_dims')

    args = parser.parse_args()
    seed_everything(args.seed)


    # Custom progress bar
    class MyTQDMProgressBar(TQDMProgressBar):
        def __init__(self):
            super(MyTQDMProgressBar, self).__init__()

        def init_validation_tqdm(self):
            bar = Tqdm(
                desc=self.validation_description,
                position=0,
                disable=self.is_disabled,
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout,
            )
            return bar


    # Use default parameters for training
    print("Training with default parameters")
    data_module = MILRegressionDataModule(args)
    model = MILRegressionModel(args)
    print(args)
    early_stopping_callback = EarlyStopping(monitor='val_loss_epoch', mode='min', patience=15)

    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_mae',
        mode='min',
        dirpath=f'best_models/{train_dataset}/in_len:{args.in_len}/best_model_{test_dataset}_LSTD:{args.use_linear}_CMTA:{args.use_align}_Transformer{args.use_transformer}/Audio:{args.audio}_Deep:{args.deep}',
        filename='{epoch}-{avg_val_mae:.4f}-{avg_val_rmse:.4f}'
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy='ddp_find_unused_parameters_true',
        logger=CSVLogger(
            save_dir=f'logs/{train_dataset}/in_len:{args.in_len}/logs_{test_dataset}_LSTD:{args.use_linear}_CMTA:{args.use_align}_Transformer{args.use_transformer}/Audio:{args.audio}_Deep:{args.deep}'),
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, MyTQDMProgressBar()],
        log_every_n_steps=15,
    )

    # Train and test the model
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)
