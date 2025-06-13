from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from model import Critic, Generator, init_weights
from dataset import DNA_Dataset
from config import (z_dim, batch_size, features_critic, features_gen,
                    learning_rate_critic, learning_rate_gen, num_epochs, lambda_gp)
from utils import gradient_penalty, count_parameters, onehot_to_fasta

if __name__ == '__main__':
    sourcefolder = "GANs_with_DNA/Simulate"
    datafolder = "SingleMotif_10bp_AllA"
    destinationfolder = f"GANs_with_DNA/Synthetic Data/{datafolder}"
    version = 6
    try:
        os.makedirs(f"{destinationfolder}/try{version}", exist_ok = False)
    except FileExistsError:
        if os.path.exists(f"{destinationfolder}/try{version}/output.log"):
            print("Change Version Number")
            raise SystemExit

    device = torch.device('cuda')
    print("Device = ",device)

    # Setup models
    critic = Critic(features_critic).to(device)
    gen  = Generator(z_dim, features_gen).to(device)
    init_weights(critic)
    init_weights(gen)

    # Print Hyperparameters
    h = open(f"{destinationfolder}/try{version}/hyperparams.txt", 'w')
    print(f"dataset={datafolder}", f"z_dim={z_dim}", f"batch_size={batch_size}",
          f"features_critic={features_critic}", f"features_gen={features_gen}",
          f"lr_critic={learning_rate_critic}", f"lr_gen={learning_rate_gen}",
          f"num_epochs={num_epochs}", f"params_gen = {count_parameters(gen)[0]}",
          f"param_critic = {count_parameters(critic)[0]}", f"lambda_gp = {lambda_gp}",
          file=h, flush=True, sep='\n')
    h.close()

    #Setup training data
    dataset = DNA_Dataset(f"{sourcefolder}/{datafolder}.fa")
    test_dataset, train_dataset= random_split(dataset, [batch_size,len(dataset)-batch_size])
    dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle = False, num_workers = 0)
    k = len(dataloader) // 5

    # Setup trainers
    optim_critic = Adam(critic.parameters(), lr=learning_rate_critic, betas=(0.0, 0.9))
    optim_gen = Adam(gen.parameters(), lr=learning_rate_gen, betas=(0.0, 0.9))
    sched_d = ReduceLROnPlateau(optimizer=optim_critic, factor=0.2,
                                patience=5, cooldown=0, eps=1e-8)
    sched_g = ReduceLROnPlateau(optimizer=optim_gen , factor=0.2,
                                patience=5, cooldown=0, eps=1e-8)

    # Evaluation on same input latent vector
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    # fixed_noise = torch.randn((1, z_dim)).to(device) # Fixed Random Seed
    # fixed_real = test_dataset[0][0].reshape(1, 4, -1).to(device)

    f = open(f"{destinationfolder}/try{version}/output.log", 'w')
    g = open(f"{destinationfolder}/try{version}/Saved.fasta", 'w')

    # Training Loop
    print("Start Training")
    # ==========================================================================
    for epoch in range(num_epochs):
        gen.train()
        critic.train()
        for batch_index , (real, _ ) in enumerate(dataloader):
            real = real.to(device)
            z = torch.randn((batch_size, z_dim)).to(device)
            fake = gen(z)

            # Train  Critic
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = torch.mean(critic(fake))  - torch.mean(critic(real)) + lambda_gp*gp

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optim_critic.step()

            # Train Generator
            critic_fake = critic(fake).reshape(-1)
            loss_gen = - torch.mean(critic_fake)
            gen.zero_grad()
            loss_gen.backward()
            optim_gen.step()

            if (len(dataloader)-1)  % k == batch_index % k:
                print(f"{datetime.now()}",
                      f"Epoch:{epoch+1}/{num_epochs}",
                      f"Batch:{batch_index+1}/{len(dataloader)}",
                      f"Loss_critic:{loss_critic:.4f}",
                      f"loss_gen:{loss_gen:.4f}",
                      sep='\t', file = f, flush=True)

        sched_d.step(loss_critic+loss_gen)
        sched_g.step(loss_critic+loss_gen)
        print(f"lr_d={sched_d.get_last_lr()[0]:.1e}",
              f"lr_gen={sched_g.get_last_lr()[0]:.1e}",
              file=f, flush=True, sep='\t')

        gen.eval()
        critic.eval()

        eval_fake = gen(torch.randn((1, z_dim)).to(device))
        eval_fake_seq = onehot_to_fasta(eval_fake.detach()[0].to('cpu'))
        print(f">{epoch}\n{eval_fake_seq}", file=g, flush=True)
        print(f"Saved Sequence{epoch+1}",
              f"Fake Prob = {torch.mean(critic(eval_fake)).item():.4f}",
              f"Real Prob = {torch.mean(critic(real)).item():.4f}")

    torch.save(gen.state_dict(), f"{destinationfolder}/try{version}/final_weights_gen.pth")
    torch.save(critic.state_dict(), f"{destinationfolder}/try{version}/final_weights_critic.pth")
    print("Training Complete")

    print(f"Fake Sequence Probability = {torch.mean(critic(eval_fake)).item():.4f}",
    f"Real Sequence Probability = {torch.mean(critic(real)).item():.4f}",
    file=f, flush=True)
    f.close()
    g.close()
