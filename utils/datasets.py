import torch
import pickle
import pandas as pd
import os
from torch.utils.data import Dataset

class WSIDataset(Dataset):
    def __init__(self, 
                 img_dir, 
                 data_file: str = None, 
                 label_cols = None,
                 file_col = None,
                 split_col: str = None,
                 split_val = None,
                 sheet_name: str = None,
                 file_ext = '.svs',
                 data_without_ext: bool = True,
                 drop_na: bool = True
                 ):
        
        self.img_dir = img_dir

        filepaths = []
        for root, dir, files in os.walk(img_dir):
            for file in files:
                filepaths.append(os.path.join(root, file))
        # filter for correct extension
        filepaths = [path for path in filepaths if os.path.splitext(path)[-1] == file_ext]
        # print(len(filepaths))
        # filepaths = os.listdir(img_dir)
        # filter filepaths for those in
        # both the data tsv and image dir, if tsv provided
        
        if data_file:
            data_ext = os.path.splitext(data_file)[-1]
            if data_ext == '.tsv':
                data_df = pd.read_csv(data_file, sep = '\t')
            elif data_ext == '.csv':
                data_df = pd.read_csv(data_file)
            elif data_ext == '.xlsx':
                if sheet_name:
                    data_df = pd.read_excel(data_file, sheet_name=sheet_name)
                else:
                    data_df = pd.read_excel(data_file)
            else:
                raise Exception(f'Unrecognised data file with extension {data_ext} - please use tsv, csv or xlsx files')
            
            # filter for data with non-nan values in label cols
            if label_cols:
                if drop_na:
                    data_df = data_df.dropna(subset=label_cols, inplace = False)
                self.label_cols = label_cols
            
            # if particular column for splitting data - i.e. train vs val, split on
            if split_col:
                data_df = data_df[data_df[split_col] == split_val]

            self.file_col = file_col
            # print(len(data_df))
        
            # remove file extensions if not part of filename in data df to match up
            if data_without_ext:
                # print(data_df[file_col], filepaths)

                filt_files = [file for file in filepaths if 
                        os.path.split(os.path.splitext(file)[0])[-1] in list(data_df[file_col].astype(str))]
            
            else:
                file_files = [file for file in filepaths if os.path.split(file)[-1] in data_df[file_col]]

            print(len(filt_files),'files in',img_dir)
            
            self.data_df = data_df[data_df[file_col].isin(filt_files)]
            self.filepaths = filt_files
            
        else:
            self.filepaths = filepaths

    def __len__(self):
        print(len(self.filepaths), 'files to process')
        return len(self.filepaths)

    def __getitem__(self, idx):
        file = self.filepaths[idx]
        # print(self.img_dir, file)
        filepath = os.path.join(self.img_dir, file) 
        
        # if label columns have been requested
        try:
            labels = self.label_cols
            return filepath, list(self.data_df.iloc[idx][labels])
        
        except:
            return filepath



class FeatureDataset(Dataset):
    def __init__(self, 
                 feature_dir: str, 
                 data_file: str, 
                 file_col: str, 
                 label_cols: list,
                 file_ext: str = '.pt',
                 data_without_ext: bool = True,
                 features_extract: int = 100,
                 drop_na: bool = True,
                 sheet_name: str = None,
                 split_col: str = None,
                 split_val = None
                 ):
        
        self.feature_dir = feature_dir
        self.file_ext = file_ext
        self.features_extract = features_extract
        
        # filter filepaths for those in
        # both the data tsv and image dir
        data_ext = os.path.splitext(data_file)[-1]
        if data_ext == '.tsv':
            data_df = pd.read_csv(data_file, sep = '\t')
        elif data_ext == '.csv':
            data_df = pd.read_csv(data_file)
        elif data_ext == '.xlsx':
            if sheet_name:
                data_df = pd.read_excel(data_file, sheet_name=sheet_name)
            else:
                data_df = pd.read_excel(data_file)
        else:
            raise Exception(f'Unrecognised data file with extension {data_ext} - please use tsv, csv or xlsx files')
        
        # filter for data with non-nan values in label cols
        if drop_na:
            data_df = data_df.dropna(subset=label_cols, inplace = False)
        
        # if particular column for splitting data - i.e. train vs val, split on
        if split_col:
            data_df = data_df[data_df[split_col] == split_val]

        self.file_col = file_col
    
        filepaths = os.listdir(feature_dir)
        # remove file extensions if not part of filename in data df
        if data_without_ext:
            filepaths = [filepath.replace(file_ext, '') for filepath in filepaths]
        filt_files = [file for file in data_df[file_col] if str(file) in filepaths]

        # print(len(filt_files),'files in',img_dir)
        
        self.data_df = data_df[data_df[file_col].isin(filt_files)]
        self.label_cols = label_cols

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        data = self.data_df.iloc[idx]
        # retrieve filepath for image
        file = str(data[self.file_col]) + self.file_ext
        filepath = os.path.join(self.feature_dir, file) 
        
        try:
            # depending on filetype, parse
            if self.file_ext == '.pt':
                features_all = torch.load(filepath)

            elif self.file_ext == '.pkl':
                with open(filepath, 'rb') as data_file:
                    features_all = pickle.load(data_file)
            
            else:
                raise Exception(f'''Unrecognised input file extension: 
                                {self.file_ext}\nPlease use either ".pt" or ".pkl" files''')
        
        except:
            raise Exception(f'Could not open file {filepath}')
        
        labels = torch.Tensor(data[self.label_cols])

        # extract required number of features
        indices = torch.randperm(len(features_all))[:self.features_extract]

        features = features_all[indices]

        return features, labels