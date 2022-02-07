import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, num_classes : int = 10) -> None:
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=0)
        self.res1 = ResidualBlock(input_dims=128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.res2 = ResidualBlock(input_dims=512)
        
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
                          
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size = (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size = (2, 2))
        x = self.res1(x)
        
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size = (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size = (2, 2))
        x = self.res2(x)

        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, input_dims: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dims, input_dims, kernel_size=(3, 3), stride=1, padding=1)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv1(out))
        return out + x
    
if __name__ == "__main__":
    rand_inp = torch.rand(3, 3, 32, 32)
    # model = ResidualBlock(3)
    model = ResNet()
    model(rand_inp)