from datetime import datetime
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from model import Discriminator, Generator, init_weights
from dataset import DNA_Dataset
from config import (z_dim, batch_size, features_disc, features_gen,
                    learning_rate_disc, learning_rate_gen, num_epochs,
                    leaky_relu_disc, leaky_relu_gen, motif,
                    max_pool_size, dropout_factor)

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params

def onehot_to_fasta(matrix):
    # print(matrix)
    seq = ""
    code = ["A", "T", "C", "G"]
    for i in matrix:
        seq = seq+code[torch.argmax(i)]
    seq = "\n".join([seq[i:i+80] for i in range(0, len(seq), 80)])
    return seq

if __name__ == '__main__':
    data = "Simulated Data"
    version = 1
    try:
        os.makedirs(f"GANs_with_DNA/{data}/logs/try{version}", exist_ok = False)
    except FileExistsError:
        if os.path.exists(f"GANs_with_DNA/{data}/logs/try{version}/output.log"):
            print("Change Version Number")
            raise SystemExit

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device = ",device)

    # Setup models
    disc = Discriminator(features_disc).to(device)
    gen  = Generator(z_dim, features_gen).to(device)
    init_weights(disc)
    init_weights(gen)

    # Print Hyperparameters
    h = open(f"GANs_with_DNA/{data}/logs/try{version}/hyperparams.txt", 'w')
    print(f"z_dim={z_dim}", f"batch_size={batch_size}",
          f"features_disc={features_disc}", f"features_gen={features_gen}",
          f"lr_disc={learning_rate_disc}", f"lr_gen={learning_rate_gen}",
          f"num_epochs={num_epochs}", f"leaky_relu_disc={leaky_relu_disc}",
          f"leaky_relu_gen={leaky_relu_gen}", f"max_pool_size={max_pool_size}",
          f"dropout_factor={dropout_factor}",
          f"params_gen = {count_parameters(gen)[0]}",
          f"param_disc = {count_parameters(disc)[0]}",
          file=h, flush=True, sep='\n')
    h.close()

    #Setup training data
    dataset = DNA_Dataset(motif, 'test.pos')
    test_dataset, train_dataset= random_split(dataset, [1,len(dataset)-1])
    dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle = False, num_workers = 0)
    k = len(dataloader) // 5

    # Setup trainers
    optim_disc = Adam(disc.parameters(), lr=learning_rate_disc, betas=(0.5,0.999))
    optim_gen = Adam(gen.parameters(), lr=learning_rate_gen, betas=(0.5,0.999))
    criterion = nn.BCELoss()
    sched_d = ReduceLROnPlateau(optimizer=optim_disc, factor=0.2,
                                patience=5, cooldown=0, eps=1e-8)
    sched_g = ReduceLROnPlateau(optimizer=optim_gen , factor=0.2,
                                patience=5, cooldown=0, eps=1e-8)

    # Evaluation on same input latent vector
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device) # Fixed Random Seed
    fixed_real = test_dataset[0][0].reshape(1, 1,4,-1)
    f = open(f"GANs_with_DNA/{data}/logs/try{version}/output.log", 'w')
    g = open(f"GANs_with_DNA/{data}/logs/try{version}/Saved.fasta", 'w')

    # print(f"LR_Gen = {learning_rate_gen}", f"LR_disc = {learning_rate_disc}",
    #       f"latent_dim = {z_dim}", f"batch_size = {batch_size}",
    #       f"features_d = {features_disc}", f"features_g = {features_gen}",
    #       f"params_gen = {count_parameters(gen)[0]}",
    #       f"param_disc = {count_parameters(disc)[0]}",
    #       sep="\t", file=f, flush=True)

    # f.close(); g.close(); raise SystemExit

    print(f"Fake Sequence Probability = {disc(gen(fixed_noise.to(device))).item()}")
    print(f"Real Sequence Probability = {disc(fixed_real.to(device)).item()}")

    # Training Loop
    print("Start Training")
    gen.train()
    disc.train()

    for epoch in range(num_epochs):
        for batch_index , (real, _ ) in enumerate(dataloader):
            real = real.reshape(batch_size, 1, 4, -1).to(device)
            z = torch.randn((batch_size, z_dim)).to(device)
            fake = gen(z)

            # Train  Discriminator
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake)/2

            disc.zero_grad()
            loss_disc.backward(retain_graph = True)
            optim_disc.step()

            # Train Generator
            disc_fake = disc(fake).reshape(-1)
            loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))
            gen.zero_grad()
            loss_gen.backward()
            optim_gen.step()

            if batch_index % k == k-1:
                print(f"{datetime.now()}",
                      f"Epoch:{epoch+1}/{num_epochs}",
                      f"Batch:{batch_index+1}/{len(dataloader)}",
                      f"Loss_disc:{loss_disc:.4f}",
                      f"loss_gen:{loss_gen:.4f}",
                      sep='\t', file = f, flush=True)

        fixed_fake = gen(fixed_noise).to(torch.device('cpu'))
        fake_seq = onehot_to_fasta(fixed_fake[0][0].T)
        print(f">{epoch}\n{fake_seq}", file=g, flush=True)
        print(f"Saved Sequence{epoch+1}")

    disc.eval()
    gen.eval()

    print(f"Fake Sequence Probability = {disc(gen(fixed_noise.to(device))).item()}",
    f"Real Sequence Probability = {disc(real.to(device)).item()}", file=f, flush=True)

    f.close()
    g.close()
    torch.save(gen.state_dict(), f"GANs_with_DNA/{data}/logs/try{version}/final_weights_gen.pth")
    torch.save(disc.state_dict(), f"GANs_with_DNA/{data}/logs/try{version}/final_weights_disc.pth")
    print("Training Complete")

    print(f"Fake Sequence Probability = {disc(gen(fixed_noise.to(device))).item()}")
    print(f"Real Sequence Probability = {disc(fixed_real.to(device)).item()}")
