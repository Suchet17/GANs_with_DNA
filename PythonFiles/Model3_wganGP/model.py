import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
# For 1000 bp DNA

class Critic(nn.Module):
    def __init__(self, features=32, num_classes=1):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv1d(4, features, kernel_size=10, padding = "same"),
            nn.ReLU(),
            nn.LayerNorm((features, 1000)),
            nn.Conv1d(features, features, kernel_size=10, padding = "same"),
            nn.ReLU(),
            nn.LayerNorm((features, 1000)),
            nn.MaxPool1d(8, padding=2),
            nn.Dropout(0.3),

            nn.Conv1d(features, features, kernel_size=10, padding = "same"),
            nn.ReLU(),
            nn.LayerNorm((features, 125)),
            nn.Conv1d(features, features, kernel_size=10, padding = "same"),
            nn.ReLU(),
            nn.LayerNorm((features, 125)),
            nn.MaxPool1d(8, padding=2),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(16*features, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.critic(x)


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
    noise_dim = 100
    num_classes = 2
    x = torch.randn((batch_size, H, W)).to(device)
    critic = Critic(57, num_classes).to(device)
    print(x.shape)
    print(critic(x).shape)
    assert critic(x).shape == (batch_size, num_classes), "Critic test failed"

    gen = Generator(noise_dim, 5)
    init_weights(gen)
    z = torch.randn((batch_size, noise_dim))
    print(z.shape)
    print(gen(z).shape)
    assert gen(z).shape == (batch_size, 4, 1000), "Generator test failed"
    print("Success, all tests passed!")

if __name__ == "__main__":
    test()
