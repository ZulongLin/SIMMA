import os
import numpy as np
import pandas as pd
import torchaudio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import trange, tqdm
import lightning as pl
import glob

import numpy as np

np.set_printoptions(suppress=True)


# Handles scientific notation strings in a numpy array.
def process_scientific_notation(matrix):
    result = np.empty_like(matrix)
    for index, value in np.ndenumerate(matrix):
        # Check if the element is a specific string representation to be converted to zero.
        if isinstance(value, str) and '-' == value.lower():
            print(value)
            result[index] = 0
        else:
            result[index] = value
    return result


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def get_from_index(labeldata, data):
    mask = (labeldata['dataname'] == data[0].split('-')[0]) & (labeldata['group'] == data[0].split('-')[1])
    result = labeldata[mask]
    return result


def subtract_median(arr):
    median = np.median(arr, axis=0)
    return arr - median


# Adjusts the number of columns in the matrix to match the target dimensions.
def process_matrix(matrix, dims):
    original_cols = matrix.shape[1]
    # Pad with zeros if the matrix has fewer columns than required.
    if original_cols < dims:
        pad_cols = dims - original_cols
        matrix = np.pad(matrix, ((0, 0), (0, pad_cols)), mode='constant')
    # Truncate if the matrix has more columns than required.
    elif original_cols > dims:
        matrix = matrix[:, :dims]

    return np.array(matrix, dtype=np.float32)


def process_matrix_V2(matrix, dims):
    original_cols = matrix.shape[1]
    # Pad by duplicating existing columns if the matrix is too narrow.
    if original_cols < dims:
        pad_cols = dims - original_cols
        fill_data = matrix[:, :original_cols]
        while fill_data.shape[1] < pad_cols:
            fill_data = np.hstack([fill_data, matrix[:, :original_cols]])
        matrix = np.hstack([matrix, fill_data[:, :pad_cols]])
    # Truncate if the matrix has more columns than required.
    elif original_cols > dims:
        matrix = matrix[:, :dims]
    return matrix


def readmulti_csv(csvfiles):
    """
    Reads multiple CSV files and concatenates them vertically.

    Parameters:
    - csvfiles (list): A list of paths to multiple CSV files.

    Returns:
    - data (pd.DataFrame): A DataFrame containing the data from all CSV files.
    """
    data = pd.DataFrame()
    for csvfile in csvfiles:
        df = pd.read_csv(csvfile)
        data = pd.concat([data, df], axis=0, ignore_index=True)
    return data


class MILDataset(Dataset):
    def __init__(self, args, labeldata, data, stage='train'):
        """
        Custom dataset class.

        Parameters:
        - args: An object containing command-line arguments.
        - labeldata (pd.DataFrame): DataFrame containing the label data.
        - data: Identifier for the data subset.
        - stage: The dataset stage ('train', 'val', 'test').
        """
        self.data_dir = args.data_dir
        self.args = args
        self.labeldata = labeldata
        self.frame_interval = args.frame_interval
        self.resultdata = []
        self.lablel = []
        self.filenames = []
        self.names = []
        self.norm = Normalizer(mean=0, std=1)
        self.input_dims = 0
        self.stage = stage
        result = get_from_index(labeldata, data).values

        seg_len = self.args.seg_len
        # Only use seg_len during the training stage.
        if seg_len is not None and self.stage == 'train':
            seg_start, seg_end = seg_len

        for i in trange(result.shape[0]):
            label_score = labeldata[labeldata['filename'] == result[i][3]]['lablescore'].values[0]
            self.lablel.append(label_score)
            self.filenames.append(result[i][3])

            # Determine filename format based on arguments.
            if args.join == 1 and 'AVEC2014' in args.train_data[0]:
                filename = result[i][3][:5]
            elif 'AVEC2017' in args.train_data[0] or 'AVEC2019' in args.train_data[0]:
                filename = result[i][3]
            else:
                filename = os.path.splitext(result[i][3])[0]

            # Skip if the file has already been processed.
            if filename in self.names:
                continue
            self.names.append(filename)

            concatenated_data = pd.DataFrame()
            min_len = float('inf')

            # Process deep features
            if args.deep:
                csvfiles = sorted(
                    glob.glob(os.path.join(args.data_dir, result[i][0] + "_data", args.deep_type, filename + '*.csv')))
                deep_data = readmulti_csv(csvfiles)
                if 'AVEC2017' in args.train_data[0]:
                    deep_columns = [col for col in deep_data.columns if
                                    ("x" in col) or ('y' in col) or ('z' in col)]
                    deep_data = deep_data[deep_columns]

                if seg_len is not None and self.stage == 'train':
                    deep_data = deep_data.iloc[seg_start:seg_end + 1]

                min_len = min(min_len, deep_data.shape[0])
                concatenated_data = pd.concat([concatenated_data, deep_data[:min_len]], axis=1)
                args.video_dims = deep_data.shape[1]

            if args.audio:
                csvfiles = sorted(
                    glob.glob(os.path.join(args.data_dir, result[i][0] + "_data", args.audio_type, filename + '*.csv')))
                audio_data = readmulti_csv(csvfiles)
                if ('AVEC2019' in self.args.train_data[0]):
                    audio_data = audio_data.iloc[:, 1:]

                if seg_len is not None and self.stage == 'train':
                    audio_data = audio_data.iloc[seg_start:seg_end + 1]

                min_len = min(min_len, audio_data.shape[0])
                concatenated_data = pd.concat([concatenated_data, audio_data[:min_len]], axis=1)
                args.audio_dims = audio_data.shape[1]

            if args.au:
                csvfiles = sorted(
                    glob.glob(
                        os.path.join(args.data_dir, result[i][0] + "_data", args.openface_type, filename + '*.csv')))
                openface_data = readmulti_csv(csvfiles)
                au_columns = [col for col in openface_data.columns if
                              ('AU' in col and "_r" in col) or ('gaze' in col and 'angle' not in col) or (
                                      'pose' in col)]
                openface_data = openface_data[au_columns]

                if seg_len is not None and self.stage == 'train':
                    openface_data = openface_data.iloc[seg_start:seg_end + 1]

                min_len = min(min_len, openface_data.shape[0])
                concatenated_data = pd.concat([concatenated_data, openface_data[:min_len]], axis=1)
                args.au_dims = openface_data.shape[1]

            if args.rppg:
                csvfiles = sorted(
                    glob.glob(os.path.join(args.data_dir, result[i][0] + "_data", args.rppg_type, filename + '*.csv')))
                if len(csvfiles) == 0:
                    continue
                rppg_data = readmulti_csv(csvfiles).iloc[1:, 1:]
                rppg_data = rppg_data.apply(lambda x: x.fillna(x.mean()), axis=1)

                if seg_len is not None and self.stage == 'train':
                    rppg_data = rppg_data.iloc[seg_start:seg_end + 1]

                min_len = min(min_len, rppg_data.shape[0])
                concatenated_data = pd.concat(
                    [concatenated_data, rppg_data.iloc[:min_len].fillna(0)], axis=1)
                args.rppg_dims = rppg_data.shape[1]

            concatenated_data.fillna(0, inplace=True)
            self.args.num_modals = sum([args.deep, args.audio, args.rppg, args.au])
            self.resultdata.append(concatenated_data)
            self.args.input_dims = self.resultdata[0].values.shape[1]
            self.lens = args.in_len

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.resultdata)

    def __getitem__(self, idx):
        data = self.resultdata[idx].values.T
        lable = self.lablel[idx]
        file_name = self.filenames[idx]

        # Pad or truncate data to a fixed length.
        data = process_matrix(data, self.lens).T
        data = data[::self.args.frame_interval]
        data = torch.tensor(data).float()

        return data, torch.tensor(lable).float()


class MILRegressionDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.frame_interval = args.frame_interval
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.input_dims = 0
        self.labeldata = pd.read_csv(args.label_file)

        self.train_dataset = MILDataset(args, self.labeldata, args.train_data, stage='train')
        self.input_dims = self.train_dataset.input_dims
        self.val_dataset = MILDataset(args, self.labeldata, args.val_data, stage='val')
        self.test_dataset = MILDataset(args, self.labeldata, args.test_data, stage='test')
        self.save_hyperparameters(args)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size,
                                 num_workers=self.num_workers)
        return test_loader
