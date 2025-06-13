import torch

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params


def onehot_to_fasta(matrix):
    # print(matrix)
    seq = ""
    code = "ACGT"
    for i in matrix.T:
        seq = seq+code[torch.argmax(i)]
    seq = "\n".join([seq[i:i+80] for i in range(0, len(seq), 80)])
    return seq


def seq_to_one_hot(seq):
    seq = seq.replace("\'", '')
    seq = seq.replace("\n", '')


    code = {"A":[1.0, 0.0, 0.0, 0.0],
            "C":[0.0, 1.0, 0.0, 0.0],
            "G":[0.0, 0.0, 1.0, 0.0],
            "T":[0.0, 0.0, 0.0, 1.0]}
    mat = [code[i] for i in seq]
    return torch.tensor(mat).T


def gradient_penalty(critic, real, fake, device = torch.device('cuda')):
    batch_size, c, l = real.shape
    epsilon = torch.rand((batch_size, 1, 1)).repeat(1, c, l).to(device)
    interpolation = fake*(1-epsilon) + real*epsilon

    interpolation_scores = critic(interpolation)
    gradient = torch.autograd.grad(inputs=interpolation, outputs=interpolation_scores,
                                   grad_outputs=torch.ones_like(interpolation_scores),
                                   create_graph=True, retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(p=2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty
