import pandas as pd
from torch.utils.data import Dataset


class CNNLoader(Dataset):

    def __init__(self, path_to_csv):
        df = pd.read_csv(path_to_csv, nrows=5)
        self.data_tuples = [(r['Original'], r['Summary']) for _, r in df.iterrows()]

    def __getitem__(self, item):
        return self.data_tuples[item]

    def __len__(self):
        return len(self.data_tuples)