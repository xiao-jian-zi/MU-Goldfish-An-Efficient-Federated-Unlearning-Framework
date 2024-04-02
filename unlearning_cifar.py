import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
# import distillation_loss
from torch.nn import functional as F
from torch.utils.data import Subset
from torch.autograd import Variable

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./checkpoints/cifar', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0
args = parser.parse_args()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

full_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize,
]), download=True)

# Proportion of data to forget
forget_Proportion = 6

split1_indices = list(range(0, len(full_dataset) // 100 * (100 - forget_Proportion)))
split2_indices = list(range(len(full_dataset) // 100 * (100 - forget_Proportion), len(full_dataset)))
train_subset1 = torch.utils.data.Subset(full_dataset, split1_indices)
train_subset2 = torch.utils.data.Subset(full_dataset, split2_indices)

'''backdoor'''
train_subset2 = list(train_subset2)
for i in range(len(train_subset2)):
    train_subset2[i] = list(train_subset2[i])
    if train_subset2[i][1] ==9:
        train_subset2[i][1] = 9
        # Embed backdoor trigger
'''backdoor'''


train_loader = torch.utils.data.DataLoader(full_dataset,
                                           batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True
                                           )
train_loader1 = torch.utils.data.DataLoader(train_subset1,
                                           batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True
                                           )

train_loader2 = torch.utils.data.DataLoader(train_subset2,
                                           batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True
                                           )

val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    normalize,
]))

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=128, shuffle=False,
    num_workers=args.workers, pin_memory=True)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
def main():
    global args, best_prec1
    args = parser.parse_args()
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()
    
    teacher_model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    teacher_model.cuda()
    
    teacher_checkpoint = torch.load('./pytorch_resnet_cifar10-master/pretrained_models/resnet32-d509ac18.th')
    teacher_model.load_state_dict(teacher_checkpoint['state_dict'])
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.NLLLoss().cuda()
    # criterion = FocalLoss(alpha=1, gamma=2).cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        '''遗忘集训练'''
        train2(train_loader2, model, optimizer, epoch)
        '''剩余集训练'''
        train1(train_loader1, model,teacher_model, criterion, optimizer, epoch)
        # train(train_loader1, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and (epoch+1) % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
def train1(train_loader, model,teacher_model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        teacher_output = teacher_model(input_var)
        # loss = criterion(output, target_var)
        loss = distillation_loss2(output, teacher_output, target_var, T=3, alpha=0.5)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            
def train2(train_loader, model,  optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = distillation_loss_unlearning(output, labels = target_var, alpha=0.25)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))   
            
def distillation_loss2(outputs,teacher_outputs,labels,T,alpha):
    # hard_loss = nn.NLLLoss().cuda()
    hard_loss = nn.CrossEntropyLoss().cuda()
    soft_loss=nn.KLDivLoss(reduction="batchmean")
    '''.log()将概率变为对数域进行计算'''
    eps=1e-6
    ditillation_loss=(soft_loss((F.softmax(outputs/T,dim=1)+eps).log(),F.softmax(teacher_outputs/T,dim=1))+
                      soft_loss((F.softmax(teacher_outputs/T,dim=1)+eps).log(),F.softmax(outputs/T,dim=1)))/2
    
    outputs = F.log_softmax(outputs, dim=1)  # 使用LogSoftmax NLL
    loss = hard_loss(outputs,labels)
    
    total_loss = loss*(1-alpha) + ditillation_loss*alpha
    return total_loss

def distillation_loss_unlearning(outputs,labels,alpha):
    soft_loss=nn.KLDivLoss(reduction="batchmean")
    fake_outputs_array = np.array([0.1*np.ones(10) for _ in range(len(outputs))])
    for i in range (len(outputs)):
        fake_outputs_array[i][labels[i]] = 1
        
    fake_outputs = torch.tensor(fake_outputs_array).cuda().float()
    loss = (soft_loss(F.softmax(outputs/1,dim=1),F.softmax(fake_outputs/0.1,dim=1))+
            soft_loss(F.softmax(fake_outputs/0.1,dim=1),F.softmax(outputs/1,dim=1)))/2
    var =0
    for i in range(len(outputs)):
        var=torch.var(outputs[i])+var
    var = (var / len(outputs))**0.1
    total_loss = -0.1*loss  + var*alpha
      
    return total_loss

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


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
    print(correct,total)
    # print('Accuracy of the network on the 10000 test images: %.2f %%' % (100*correct/total))
    print('%.2f %%'%(100*correct/total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    sumnot7but7 = 0
    for data in testloader:
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        labels = labels.cuda()
        outputs = net(Variable(images).cuda())
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cuda()
        c = (predicted == labels).squeeze()
        # print(predicted)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
            if label== 9 and predicted[i]==1:
                sumnot7but7 += 1 
    for i in range(10):
        if class_total[i] != 0:
            # print(class_total[i])
            prin_tresult = 100 * class_correct[i] / class_total[i]
        else:
            prin_tresult = 0
        # print('Accuracy of %5s : %.2f %%' % (classes[i], prin_tresult))
        print('%.2f %%' %  prin_tresult)
    print('backdoor_acc：',sumnot7but7/class_total[9],class_total[9])
    
def backdoor_acc_test(path,testloader,val_loader):
    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    test(testloader, model)
    criterion = nn.CrossEntropyLoss().cuda()
    validate(val_loader, model, criterion)

if __name__ == '__main__':
    main()
    
    
    
