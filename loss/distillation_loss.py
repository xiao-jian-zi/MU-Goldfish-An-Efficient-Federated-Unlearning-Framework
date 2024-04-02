import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def distillation_loss1(outputs,teacher_outputs,labels,T,alpha):
    # loss=hard_loss(out,label)
    # ditillation_loss=soft_loss(F.softmax(out/T,dim=1),F.softmax(teacher_output/T,dim=1))
    # loss_all=loss*alpha+ditillation_loss*(1-alpha)
    hard_loss=nn.CrossEntropyLoss()
    soft_loss=nn.KLDivLoss(reduction="batchmean")
    loss = hard_loss(outputs,labels)
    '''.log()将概率变为对数域进行计算'''
    eps=1e-6
    ditillation_loss=(soft_loss((F.softmax(outputs/T,dim=1)+eps).log(),F.softmax(teacher_outputs/T,dim=1))+
                      soft_loss((F.softmax(teacher_outputs/T,dim=1)+eps).log(),F.softmax(outputs/T,dim=1)))/2
    total_loss = loss + ditillation_loss*alpha
    return total_loss

def distillation_loss2(outputs,teacher_outputs,labels,T,alpha):
    # loss=hard_loss(out,label)
    # ditillation_loss=soft_loss(F.softmax(out/T,dim=1),F.softmax(teacher_output/T,dim=1))
    # loss_all=loss*alpha+ditillation_loss*(1-alpha)
    hard_loss=nn.CrossEntropyLoss().cuda()
    soft_loss=nn.KLDivLoss(reduction="batchmean")
    '''.log()将概率变为对数域进行计算'''
    ditillation_loss=soft_loss(F.softmax(outputs/T,dim=1),F.softmax(teacher_outputs/T,dim=1))
    loss = hard_loss(outputs,labels)
    total_loss = loss + ditillation_loss*alpha
    return total_loss

def distillation_loss_unlearning(outputs,labels,alpha):
    # hard_loss=nn.CrossEntropyLoss().cuda()
    soft_loss=nn.KLDivLoss(reduction="batchmean").cuda()
    fake_outputs_array = np.array([0.1*np.ones(10) for _ in range(len(outputs))])
    for i in range (len(outputs)):
        fake_outputs_array[i][labels[i]] = 1
    fake_outputs = torch.tensor(fake_outputs_array).cuda().float()
    loss = (soft_loss(F.softmax(outputs/1,dim=1),F.softmax(fake_outputs/0.1,dim=1))+
            soft_loss(F.softmax(fake_outputs/0.1,dim=1),F.softmax(outputs/1,dim=1)))/2
    # loss = hard_loss(outputs,labels)
    var =0
    for i in range(len(outputs)):
        var=torch.var(outputs[i])+var
    var = (var / len(outputs))**0.1
    total_loss = -0.1*loss  + var*alpha
        
    return total_loss
    