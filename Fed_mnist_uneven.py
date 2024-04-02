import copy
import torch
import torchvision
import torchvision.transforms as transforms
import ssl
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from Nets import base_CNN, AlexNetCifar, ResNet, resnet34, CNNMnist, args, MLP
import pickle
from utils import ada_hessain
from loss import distillation_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import csv


def save_tensor_list_to_csv(tensor_list, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["value"])  
        for tensor in tensor_list:
            value = tensor.item()  
            writer.writerow([value])  


ssl._create_default_https_context = ssl._create_unverified_context
'''# minst'''
args.num_channels = 1
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_dataset_train = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
mnist_dataset_test = torchvision.datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)


clients_nums = 5
split_size = 2000
# indices = torch.randperm(len(mnist_dataset_train)).tolist()
indices = list(range(0, len(mnist_dataset_train)))
sub_datasets = []
start_idx = 0
for i in range(clients_nums):
    end_idx = start_idx + (i+1) * split_size if i < clients_nums - 1 else len(mnist_dataset_train)
    sub_dataset = torch.utils.data.Subset(mnist_dataset_train, indices[start_idx:end_idx])
    sub_dataset_dataloader = torch.utils.data.DataLoader(sub_dataset, batch_size=100, shuffle=False, num_workers=2)
    sub_datasets.append(sub_dataset_dataloader)
    start_idx += len(sub_dataset)

testloader = torch.utils.data.DataLoader(mnist_dataset_test, batch_size=100, shuffle=False, num_workers=2)
img_size = mnist_dataset_train[0][0].shape
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

net_test = CNNMnist(args)

nets = [CNNMnist(args).cuda() for _ in range(clients_nums)]
for net in nets:
    net.load_state_dict(net_test.state_dict())

optimizers = [optim.SGD(nets[k].parameters(), lr=0.001, momentum=0.9, weight_decay=0) for k in range(clients_nums)]

criterion = nn.CrossEntropyLoss()

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test(testloader, net):
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
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

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


def Accuracy(testloader, net):
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
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return 100 * correct / total


def FedAvg(nets_list):
    # 求和
    w_avg = copy.deepcopy(nets_list[0].state_dict())
    # state_dict()
    for k in w_avg.keys():
        '''print(k)
        layer_input.weight
        layer_input.bias
        layer_hidden.weight
        layer_hidden.bias
        '''
        for i in range(1, len(nets_list)):
            w_avg[k] += nets_list[i].state_dict()[k]
        w_avg[k] = torch.div(w_avg[k], len(nets_list))
    return w_avg

def FedAvg2(nets_list,clients_acc_list):
    # 求和
    w_avg = copy.deepcopy(nets_list[0].state_dict())
    mean = sum(clients_acc_list)/len(clients_acc_list)

    for i in range(len(clients_acc_list)):
        clients_acc_list[i] = math.exp(-(mean - clients_acc_list[i])/mean)
        
    sum_ = sum(clients_acc_list)
    
    for i in range(len(clients_acc_list)):
        clients_acc_list[i] = clients_acc_list[i]/sum_
        if i == len(clients_acc_list)-1:
            clients_acc_list[i] = 0
            clients_acc_list[i] = 1-sum(clients_acc_list)

    for k in w_avg.keys():
        for i in range(1, len(nets_list)):
            w_avg[k] = (torch.mul(w_avg[k],clients_acc_list[0]) 
                        + torch.mul(nets_list[i].state_dict()[k],clients_acc_list[i]))
        # w_avg[k] = torch.div(w_avg[k], len())
    return w_avg

def train(epoch_num = 10):
    loss_list = []
    Accuracy_list = []
    forget_acc_list = []
    nets_list = []
    for epoch in tqdm(range(epoch_num), desc='training', unit='epoch'):
        timestart = time.time()
        running_loss = 0.0
        clients_acc_list = [ 0 for _ in range(clients_nums)]
        '''训练集的训练'''
        for k in range(len(sub_datasets)):
            for i, data in enumerate(sub_datasets[k], 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizers[k].zero_grad()
                outputs = nets[k](inputs)
                loss = criterion(outputs, labels)
                '''T=math.exp(1-forget_Proportion/100)'''
                '''loss的输出值为：tensor(1.3511, grad_fn=<NllLossBackward0>)'''
                loss.backward()
                optimizers[k].step()
                running_loss += loss.item()
            acc_ = Accuracy(testloader, nets[k])
            clients_acc_list[k]=acc_
            
        print('[%d ,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 600))
        loss_list.append(running_loss / 600)
        running_loss = 0.0
        print('epoch %d cost %3f sec' % (epoch + 1, time.time() - timestart))
        if (max(clients_acc_list) - min(clients_acc_list))>30:
            a = FedAvg2(nets,clients_acc_list)
        else:
            a = FedAvg(nets)
        # a = FedAvg(nets)
        for net in nets:
            net.load_state_dict(a)
        # nets = [net for _ in range(num_splits)]
        print('FedAvg acc:')
        acc = Accuracy(testloader, nets[0])
        Accuracy_list.append(acc)
    print('Finished Training')
    net.load_state_dict(FedAvg(nets))
    test(testloader, net)


if __name__ == '__main__':
    train(epoch_num = 10)

