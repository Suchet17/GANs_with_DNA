import torch
from torch.utils.data import Dataset

def seq_to_one_hot(seq):
    seq = seq.replace("\'", '')
    seq = seq.replace("\n", '')


    code = {"A":[1.0, 0.0, 0.0, 0.0],
            "C":[0.0, 1.0, 0.0, 0.0],
            "G":[0.0, 0.0, 1.0, 0.0],
            "T":[0.0, 0.0, 0.0, 1.0]}
    mat = [code[i] for i in seq]
    return torch.tensor(mat).T

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

'''
dataset = DNA_Dataset("GANs_with_DNA/Simulated Data/Simulate/SingleMotif_len10_Centre150.fa")
test_dataset, train_dataset= random_split(dataset, [1,len(dataset)-1])
dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle = False, num_workers = 0)
'''
