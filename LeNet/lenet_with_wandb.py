import torch as T
from torch import nn
import torch.optim as optim
import torchvision as tv
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
import random

T.manual_seed(0)
T.cuda.manual_seed_all(0)

filename = os.path.splitext(os.path.basename(__file__))[0]
Path = f'./model/{filename}.pt'
data_path = './data'
project_name = f"{filename}"
val_ratio = 0.1
config = {
    "epochs": 10,
    "batch_size": 128,
    "lr": 1e-3,
    "momentum": 0.99,
}
device = T.device(T.accelerator.current_accelerator().type if T.accelerator.is_available() else 'cpu')

def imshow(img):
    img = img.detach().cpu()
    img = img / 2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(T.relu(self.conv1(x)))
        x = self.pool(T.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(net, trainloader, valloader, criterion, optimizer, device):
    best_acc = 0.0

    with wandb.init(project=project_name, config=config) as run:

        for epoch in range(config['epochs']):
            print(f"Epoch {epoch + 1}/{config['epochs']}")
            net.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach()
                wandb.log({"batch_loss": loss.detach()})

            epoch_loss = running_loss / len(trainloader)
            wandb.log({"epoch_loss": epoch_loss})

            val_acc = validate(net, valloader, device)
            print(f"val_acc {epoch + 1}: {val_acc:.3f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                cpu_state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()} 
                T.save(cpu_state_dict, Path)
                print(f"best model saved with acc: {best_acc:.3f}%\n")

    wandb.finish()
    print("Finished Training")

def validate(net, valloader, device):
    net.eval()
    correct = 0
    total = 0

    with T.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = T.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def test(net, testloader, classes, device):
    net.eval()
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    with T.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = T.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            labels = labels.cpu().tolist()
            predicted = predicted.cpu().tolist()
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(f'Total Accuracy: {100 * correct / total:.2f} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.2f} %')


if __name__ == '__main__':
    wandb.login()
    print('device: ', device)

    cifar10_dir = 'cifar-10-batches-py'
    required_files = [
        'data_batch_1', 'data_batch_2', 'data_batch_3',
        'data_batch_4', 'data_batch_5', 'test_batch', 'batches.meta'
    ]

    dataset_path = os.path.join(data_path, cifar10_dir)
    download = not all(os.path.exists(os.path.join(dataset_path, file)) for file in required_files)

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(T.float32, scale = True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = tv.datasets.CIFAR10(root = data_path, train = True, download = download, transform = transform)

    val_size = int(len(trainset) * val_ratio)
    train_size = len(trainset) - val_size
    trainset, valset = T.utils.data.random_split(trainset, [train_size, val_size])
    
    trainloader = T.utils.data.DataLoader(trainset, batch_size = config["batch_size"], shuffle = True, num_workers = 2, persistent_workers=True, pin_memory=(device.type == 'cuda'))
    valloader = T.utils.data.DataLoader(valset, batch_size = config["batch_size"], shuffle = False, num_workers = 2, persistent_workers=True,pin_memory=(device.type == 'cuda'))

    testset = tv.datasets.CIFAR10(root = data_path, train = False, download = download, transform = transform)
    testloader = T.utils.data.DataLoader(testset, batch_size = config["batch_size"], shuffle = False, num_workers = 2, persistent_workers=True, pin_memory=(device.type == 'cuda'))

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    net =LeNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = config["lr"], momentum = config["momentum"])

    train(net, trainloader, valloader, criterion, optimizer, device)
    net.load_state_dict(T.load(Path, map_location=device, weights_only=True))
    test(net, testloader, classes, device)