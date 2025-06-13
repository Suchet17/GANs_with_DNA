from torch.utils.data import Dataset
from utils import seq_to_one_hot

class DNA_Dataset(Dataset):
    def __init__(self, path):
        if path.endswith('.fa'):
            pass
        else:
            path = path+'.fa'
        self.path = path
        f = open(path, 'r')
        self.seqs = f.readlines()[1::2]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
            return seq_to_one_hot(self.seqs[index]), 1
