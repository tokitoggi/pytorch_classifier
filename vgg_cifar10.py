import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms

# import existed vgg file
import vgg
from vgg import VGG

cfg = { #8 + 3 =11 == vgg11 
   'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    # 10 + 3 = vgg 13 
   'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    # 13 + 3 = vgg 16 
   'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 
    # 16 +3 =vgg 19 
   'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] }

epochs = 10
batch_size = 64
learning_rate = 0.01

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

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


model = VGG(vgg.make_layers(cfg['D']),10,True).to(device)
print(model)
criterion = nn.CrossEntropyLoss().to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

print("The size of trian data is ",len(trainloader))
# for training 
print("The epochs is %d. The Batch size is %d. Learning Rate is %.5f." % (epochs, batch_size, learning_rate))
def train(model, trainloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader,0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 30 == 0:
            print('Train epoch : {} [{}/{} {:.0f}%]\tLoss:{:.6f}'.format(epoch, batch_idx*len(data),len(trainloader.dataset),100.*batch_idx/len(trainloader),loss.item()))

def evaluate(model, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = torch.max(output.data, 1)[1]
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

