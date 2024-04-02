import torch
import torchvision
import torchvision.transforms as transforms
import ssl
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from Nets import base_CNN,AlexNetCifar,ResNet,resnet34,CNNMnist,args,MLP
import pickle
from utils import ada_hessain
import csv
# Check if CUDA is available, use it if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Function to save a list of tensors to a CSV file
def save_tensor_list_to_csv(tensor_list, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["value"])  # Write column name
        for tensor in tensor_list:
            value = tensor.item()  # Extract the value of the tensor
            writer.writerow([value])  # Write the value to the CSV file

ssl._create_default_https_context = ssl._create_unverified_context
# Set number of channels for MNIST dataset
args.num_channels = 1
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_dataset_train = torchvision.datasets.FashionMNIST('./data/fmnist/', train=True, download=True, 
                                                 transform=trans_mnist)
mnist_dataset_test = torchvision.datasets.FashionMNIST('./data/fmnist/', train=False, download=True, 
                                                transform=trans_mnist)



trainloader = torch.utils.data.DataLoader(mnist_dataset_train, batch_size=100, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(mnist_dataset_test, batch_size=100, shuffle=False, num_workers=2)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# AlexNetCifar ResNet  resnet34(num_classes=10)  MLP(img_size, args)
net = CNNMnist(args)
net.to(device)
criterion = nn.CrossEntropyLoss()

# Choose optimizer
'''SGD+Momentum'''
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,weight_decay=0)
'''Adagrade'''
# optimizer = optim.Adagrad(net.parameters(), lr=0.001, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
'''RMSprop'''
# optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.99, eps=1e-08)
'''AdaDelta'''
# optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06)
'''Adam'''
# optimizer = optim.Adam(net.parameters(), lr=0.001)
'''AdamW'''
# optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
'''NAdam'''
# optimizer = optim.NAdam(net.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)
'''adahessian'''
# optimizer = ada_hessain.AdaHessian(net.parameters())

# Function to display an image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# Function to test the network's accuracy    
def test(testloader,net):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        labels = labels.cuda()
        outputs = net(Variable(images).cuda())
        # print(F.softmax(outputs,dim=1))
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cuda()
        total += labels.size(0)
        correct += (predicted == labels).sum()
    # print(correct,total)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        labels = labels.cuda()
        outputs = net(Variable(images).cuda())
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cuda()
        c = (predicted == labels).squeeze()
        # print(c)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    for i in range(10):
        if class_total[i] != 0:
            # print(class_total[i])
            prin_tresult = 100 * class_correct[i] / class_total[i]
        else:
            prin_tresult = 0
        print('Accuracy of %5s : %2d %%' % (classes[i], prin_tresult))
        
def Accuracy(testloader,net):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        labels = labels.cuda()
        outputs = net(Variable(images).cuda())
        # print(F.softmax(outputs,dim=1))
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cuda()
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))
    return 100*correct/total
# Function to train the network
def train(epoch_num = 5):
    loss_list = []
    for epoch in tqdm(range(epoch_num),desc='training',unit='epoch'):
        timestart = time.time()
        running_loss = 0.0
        '''训练集的训练'''
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
          
        print('[%d ,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 600))
        loss_list.append(running_loss / 600)
        running_loss = 0.0
        Accuracy(testloader,net)
        print('epoch %d cost %3f sec' % (epoch + 1, time.time()-timestart))
    print('Finished Training')

if __name__ == '__main__':
    train(epoch_num = 1)
    test(testloader, net)
    