import torch.nn as nn
import torch

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100,588),
            nn.ReLU(True)
        )
        self.up=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,3,stride=1,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.out = nn.Conv2d(16,3,1)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1,3,14,14)
        out = self.up(out)
        out = self.out(out)

        return out

if __name__ == '__main__':
    pass

