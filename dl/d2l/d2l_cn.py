#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

from IPython import display

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import os

#6_3 nlp lyrics

import torch
import random
import zipfile

import torch.nn.functional as F

def set_figsize(figsize=(3.5,2.5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
# 保存在3.11
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr*param.grad/batch_size#lr 学习率


def train_ch3(net, train_iter, test_iter, loss, epochs, batch_size, params=None, lr=None, optimizer=None):
    # 训练
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:  # X为图像
            y_hat = net(X)  # 对该次输入的预测值
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()

            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()  # 统计总的损失
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()  # 统计准确率
            # y.shape[0]为batch_size个label
            n += y.shape[0]
        # 一个epoch走完
        test_acc = evaluate_accuracy(test_iter, net)
        print("Epoch %d, loss %.4f, train accuracy %.3f, test accuracy %.3f" % (
        epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def load_data_from_fmnist(batch_size, resize=None):
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))

    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='/Users/bob/docs/dataset', train=True, download=False,
                                                    transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root='/Users/bob/docs/dataset/', train=False, download=False,
                                                   transform=transform)

    train_it = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_it = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_it, test_it

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)


def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y

# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进。
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))



class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
    
def load_data_jay_lyrics():
    with zipfile.ZipFile("../chapter_6/jaychou_lyrics.txt.zip") as in_data:
        with in_data.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    #歌词语料字符
    corpus_chars = corpus_chars.replace('\n',' ').replace('\r',' ')
    #索引和字符一一对应
    idx_to_char = list(set(corpus_chars))
    #字符和索引一一对应
    char_to_idx = dict([(char,i) for i,char in enumerate(idx_to_char)])
    #字符表大小
    vocab_size = len(char_to_idx)
    #corpus_chars里的每个字符在字符表里的索引
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    
    return corpus_indices,char_to_idx,idx_to_char,vocab_size


## chapter_6_3
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    #每个样本包含num_steps时间步
    #TODO: 这里为啥要除以num_steps
    num_examples = (len(corpus_indices) - 1) // num_steps#减一？//多少条样本
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))#所有的样本索引
    random.shuffle(example_indices)
    
    def _data(pos):
        return corpus_indices[pos:pos+num_steps]#返回一个样本
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    for i in range(epoch_size):
        i = i*batch_size
        batch_indices = example_indices[i : i+batch_size]#每次取example_indices[i]值代表的下标开始的样本，取batch次
        X = [_data(j*num_steps) for j in batch_indices]
        Y = [_data(j*num_steps + 1) for j in batch_indices]
        
        yield torch.tensor(X,dtype=torch.float32,device = device), torch.tensor(Y,dtype = torch.float32, device = device)
    
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if (device is None) :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    
    batch_len = data_len // batch_size
    
    indices = corpus_indices[0:batch_size * batch_len].view(batch_size,batch_len)
    
    epoch_size = (batch_len - 1) // num_steps
    
    for i in range(epoch_size):
        i = i* num_steps
        X = indices[:,i:i+num_steps]
        Y = indices[:,i+1:i+num_steps+1]
        
        yield X,Y

def one_hot(x,n_class,dtype=torch.float32):
    x = x.long()
    res = torch.zeros(x.shape[0],n_class,dtype=dtype,device=x.device)
    res.scatter_(1,x.view(-1,1),1)
    return res

def to_onehot(X,n_class):
    return [one_hot(X[:,i],n_class) for i in range(X.shape[1])]

#梯度裁剪
def grad_clipping(params, theta, device):
    norm  = torch.tensor([0.0],device = device)
    for parm in params:
        norm += (parm.grad.data**2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)
            
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, is_random_iter,num_epochs, num_steps, lr, clipping_theta, batch_size, pred_peroid, pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        #如果不是随机采样，相邻采样
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        
        data_iter = data_iter_fn(corpus_indices,batch_size,num_steps, device)
        
        for X,Y in data_iter:
            # 如使用随机采样，在每个小批量更新前初始化隐藏状态
            if is_random_iter:
                state = init_rnn_state(batch_size,num_hiddens, device)
                 # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
            # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
            else:
                for s in state:
                    s.detach_()
            inputs = to_onehot(X,vocab_size)
            (outputs, state) = rnn(inputs, state, params)
            outputs = torch.cat(outputs, dim = 0)
            y = torch.transpose(Y,0,1).contiguous().view(-1)
            l = loss(outputs, y.long())#一个批次的交叉熵一起做，计算平均分类误差
            
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
                
            l.backward()
            
            grad_clipping(params, clipping_theta, device)
            sgd(params, lr , 1)# 因为误差已经取过均值，梯度不用再做平均
            
            l_sum += l.item() * y.shape[0]
            n+=y.shape[0]
        
        if (epoch + 1) % pred_peroid == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % ( epoch+1 , math.exp(l_sum/n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                        num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
            