import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# import existed vgg file
import resnet
from resnet import ResNet

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

epochs = 1
batch_size = 64
learning_rate = 0.01

#transform 
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
# Download Cifar10
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, 
                                        download=True, transform=transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, 
                                        shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = ResNet(resnet.ResidualBlock, [3, 4, 6, 3]).to(device)
print(model)
print("The epochs is %d. The Batch size is %d. Learning Rate is %.5f." % (epochs, batch_size, learning_rate))
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

def train(model, trainloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print('step size : %d, loss : %f' % (batch_idx, loss.item()))
        if batch_idx % 100 == 0:
            print('Train epoch : {} [{}/{} {:.0f}%]\tLoss:{:.6f}'.format(epoch, batch_idx*len(data),len(trainloader.dataset),100.*batch_idx/len(trainloader),loss.item()))

def evaluate(model, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(testloader.dataset)
    test_accuracy = 100 * correct / len(testloader.dataset)
    return test_loss, test_accuracy

print("The train is started!!")
for epoch in range(1, epochs + 1):
    train(model, trainloader, optimizer, epoch)
    scheduler.step()
    test_loss, test_accuracy = evaluate(model, testloader)
    print('[{}] Test Loss : {:4f}, Accuracy : {:.2f}%'.format(epoch, test_loss, test_accuracy))
print("The train and test is finished!")
