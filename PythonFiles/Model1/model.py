import torch
import torch.nn as nn
from config import leaky_relu_gen, leaky_relu_disc, max_pool_size, dropout_factor
# For 1000 bp DNA

class Discriminator(nn.Module):
    def __init__(self, features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=features, kernel_size=(1,10), padding='same'),
            nn.Dropout(dropout_factor),
            nn.BatchNorm2d(num_features=features),
            nn.LeakyReLU(leaky_relu_disc),
            nn.MaxPool2d(kernel_size=(1, max_pool_size)),
            nn.Flatten(),
            nn.Linear(112*features, 1),
            nn.Dropout(dropout_factor),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 194*4*features),
            nn.Dropout(dropout_factor),
            nn.LeakyReLU(leaky_relu_gen),
            nn.Unflatten(1, (features, 4, -1)),
            nn.ConvTranspose2d(features, 1, kernel_size=(1,35), stride=(1,5)),
            nn.Dropout(dropout_factor),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


def init_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch_size, H, W = 2, 4, 1000
    noise_dim = 100
    x = torch.randn((batch_size, 1, H, W)).to(device)
    disc = Discriminator(2048).to(device)
    print(x.shape)
    print(disc(x).shape)
    assert disc(x).shape == (batch_size, 1), "Discriminator test failed"

    gen = Generator(100, 3)
    init_weights(gen)
    z = torch.randn((batch_size, noise_dim))
    print(z.shape)
    print(gen(z).shape)
    assert gen(z).shape == (batch_size, 1, 4, 1000), "Generator test failed"
    print("Success, tests passed!")

if __name__ == "__main__":
    test()
