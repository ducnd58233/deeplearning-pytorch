import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(
        self, 
        num_classes : int = 10
    ) -> None:
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5), stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5), stride=1, padding=0)
        self.mlp1 = nn.Linear(800, 500)
        self.mlp2 = nn.Linear(500, self.num_classes)
        
        self._init_weights
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, (2, 2)))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, (2, 2)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)
        return x
    
if __name__ == "__main__":
    rand_inp = torch.rand(3, 1, 28, 28)
    model = LeNet()
    model(rand_inp)
    print(model)