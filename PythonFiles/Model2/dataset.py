import gzip
import torch
from torch.utils.data import Dataset

def seq_to_one_hot(seq):
    seq = seq.replace("\'", '')

    code = {"A":[1.0, 0.0, 0.0, 0.0],
            "T":[0.0, 1.0, 0.0, 0.0],
            "C":[0.0, 0.0, 1.0, 0.0],
            "G":[0.0, 0.0, 0.0, 1.0]}
    mat = [code[i] for i in seq]
    return torch.tensor(mat).T

def binfasta_to_list(path):
    f = gzip.open(path)
    data = str(f.read()).strip().split('\\n')[1::2]
    f.close()
    return data


class DNA_Dataset(Dataset):
    def __init__(self, motif, x = 'test.pos'):
        # choose x from ['train.neg', 'train.pos', 'test.neg',
        #                'test.pos', 'valid.neg', 'valid.pos']
        self.x = x
        self.seqs = binfasta_to_list(f'GANs_with_DNA/Simulated Data/{motif}.{x}.fasta')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        if self.x.endswith('pos'):
            return [seq_to_one_hot(self.seqs[index]), 1]
        else:
            return [seq_to_one_hot(self.seqs[index]), 0]
