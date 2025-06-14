import torch
import torch.nn as nn
# For 1000 bp DNA

class Discriminator(nn.Module):
    def __init__(self, features=32, num_classes=1):
        super(Discriminator, self).__init__()
        kernel_size=10
        self.disc = nn.Sequential(
            nn.Conv1d(4, features, kernel_size=kernel_size, padding = "same"),
            nn.ReLU(),
            nn.Conv1d(features, features, kernel_size=kernel_size, padding = "same"),
            nn.ReLU(),
            nn.MaxPool1d(8, padding=2),
            nn.Dropout(0.3),

            nn.Conv1d(features, features, kernel_size=kernel_size, padding = "same"),
            nn.ReLU(),
            nn.Conv1d(features, features, kernel_size=kernel_size, padding = "same"),
            nn.ReLU(),
            nn.MaxPool1d(8, padding=2),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(16*features, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 32*features),

            nn.Unflatten(1, (-1, 32)),
            nn.ConvTranspose1d(features, features, kernel_size=4, stride=2, padding=2), #62
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(features, features, kernel_size=8 , stride=4, padding=1), #250
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(features, 4, kernel_size=10, stride=4, padding=3), #1000
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


def init_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.2)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0, 0.2)

def test():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch_size, H, W = 2, 4, 1000
    noise_dim = int(5e4)
    num_classes = 2
    x = torch.randn((batch_size, H, W)).to(device)
    disc = Discriminator(51, num_classes).to(device)
    print(x.shape)
    print(disc(x).shape)
    assert disc(x).shape == (batch_size, num_classes), "Discriminator test failed"

    gen = Generator(noise_dim, 5)
    init_weights(gen)
    z = torch.randn((batch_size, noise_dim))
    print(z.shape)
    print(gen(z).shape)
    assert gen(z).shape == (batch_size, 4, 1000), "Generator test failed"
    print("Success, all tests passed!")

if __name__ == "__main__":
    test()
