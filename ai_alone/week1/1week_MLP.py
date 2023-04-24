# 패키지 불러오기 
import torch 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# 하이퍼파라메터
batch_size = 100 
hidden_size = 500 
num_classes = 10
lr = 0.001
epochs = 3 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 불러오기 
# dataset 
train_dataset = MNIST(root='./mnist', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='./mnist', train=False, transform=ToTensor(), download=True)

# dataloader 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 전처리 

# 모델 class 
class myMLP(nn.Module): 
    def __init__(self, hidden_size, num_classes): 
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, c, w, h = x.shape  # 100, 1, 28, 28  
        x = x.reshape(-1, 28*28) # 100, 28x28 
        # x = x.reshape(b, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


# 모델, loss, optimizer 
model = myMLP(hidden_size, num_classes).to(device)
loss = nn.CrossEntropyLoss() 
optim = Adam(model.parameters(), lr=lr)

# 학습 loop 
for epoch in range(epochs): 
    for idx, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        
        out = model(image)
        loss_value = loss(out, target)
        optim.zero_grad() 
        loss_value.backward()
        optim.step()

        if idx % 100 == 0 : 
            print(loss_value.item())
