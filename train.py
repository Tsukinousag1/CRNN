from __future__ import print_function
from __future__ import division

import argparse
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np

#from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss

import os
import utils
import dataset

import crnn as net
import params

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', type=str, default='/home/std2021/hejiabang/OCR/CRNN/db', help='lmdb data train path')  # ../CRNN
parser.add_argument('--valroot', type=str, default='/home/std2021/hejiabang/OCR/CRNN/db', help='lmdb data val path')  # ../CRNN
args=parser.parse_args()

#如果没有存储sample和model的地方，新建一个
if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

#ensure everytime the random is the same
#设置随机种子是为了确保每次生成固定的随机数
random.seed(params.manualSeed)#1234
np.random.seed(params.manualSeed)
#为CPU中设置种子，生成随机数
torch.manual_seed(params.manualSeed)


cudnn.benchmark=True
#可以增加运行效率

if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device,so you should probably set cuda in parms.py to True")

#--------------------------------------------------------

"""
In this block 
    Get train and val data_loader
"""

def data_loader():
    #train
    train_dataset=dataset.lmdbDataset(root=args.trainroot)

    assert train_dataset

    if not params.random_sample:
        sampler=dataset.randomSequentialSampler(train_dataset,params.batchSize)
    else:
        sampler=None

    train_loader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batchSize,
        shuffle=True,
        sampler=sampler,
        num_workers=int(params.workers),
        collate_fn=dataset.alignCollate(imgH=params.imgH,imgW=params.imgW,keep_ratio=params.keep_ratio)
    )

    #val
    val_dataset=dataset.lmdbDataset(root=args.valroot,transform=dataset.resizeNormalize((params.imgH,params.imgW)))
    assert  val_dataset
    val_loader=torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params.batchSize,
        shuffle=True,
        num_workers=int(params.workers)
    )

    return train_loader,val_loader

train_loader,val_loader=data_loader()
#------------------------------------------------------------

"""
#In this block
#    Net init
#    Weight init
#    Load pretrained model
"""

def weight_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm')!=-1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

def net_init():
    nclass=len(params.alphabet)+1
    crnn=net.CRNN(params.imgH,params.nc,nclass,params.nh)
    crnn.apply(weight_init)
    if params.pretrained!='':
        print('loading pretrained model from %s' % params.pretrained)
        if params.multi_gpu:
            crnn=torch.nn.DataParallel(crnn)
        crnn.load_state_dict(torch.load(params.pretrained))
    return crnn

crnn=net_init()
#print(crnn)

#--------------------------------------------
"""
In this block
    Init some utils defined in utils.py
"""

#Compute average for 'torch.Variable' and 'torch.Tensor'.
loss_avg=utils.averager()

#Convert between str and label.
converter=utils.strLabelConverter(params.alphabet)

#--------------------------------------------------------------
"""
In this block
    criterion define
"""

criterion=CTCLoss()

#------------------------------------------------------
"""
In this block
    Init some tensor
    put tensor and net on cuda
    NOTE:
        image,text,length is used by both val and train
        because train and val will never use it at same time 
"""

#----------------------------------------------------------
#保证放入最大值
image=torch.FloatTensor(params.batchSize,3,params.imgH,params.imgH)
text=torch.LongTensor(params.batchSize*5)
length=torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion=criterion.cuda()
    image=image.cuda()
    text=text.cuda()

    crnn=crnn.cuda()

    if params.multi_gpu:
        crnn=torch.nn.DataParallel(crnn,device_ids=range(params.ngpu))

image=Variable(image)
text=Variable(text)
length=Variable(length)

#--------------------------------------------------------------
"""
In this block
    Setup optimizer
"""

if params.adam:
    optimizer=optim.Adam(crnn.parameters(),lr=params.lr,betas=(params.beta1,0.999))
elif params.adadelta:
    optimizer=optim.Adadelta(crnn.parameters())
else:
    optimizer=optim.RMSprop(crnn.parameters(),lr=params.lr)

#--------------------------------------------------------------
"""
In this block
    Dealwith lossnan
    NOTE:
        dealwith loss nan according to the torch vision.
"""

if params.dealwith_lossnan:
    criterion=CTCLoss(zero_infinity=True)
    #CTCLoss的zero_infinity代表是否将无限大的损失和梯度归零
    # 无限损失主要发生在输入太短而无法与目标对齐时。

#----------------------------------------------------------------

def train(net,criterion,optimizer,train_iter):
    for p in crnn.parameters():
        p.requires_grad=True
    crnn.train()

    data=train_iter.next()#用于返回文件下一行

    #开始时image,t,l随机生成，然后将其变为cpu_image,cpu_text的复制
    cpu_images,cpu_texts=data
    batch_size=cpu_images.size(0)
    #[10,1,100,32]
    utils.loadData(image,cpu_images)
    #复制一个得到image
    t,l=converter.encode(cpu_texts)
    #同理复制得到t,l
    utils.loadData(text,t)
    utils.loadData(length,l)

    optimizer.zero_grad()
    preds=crnn(image)#preds[0]：宽
    #preds:[26,10,37] [26*10,37]
    preds_size=Variable(torch.LongTensor([preds.size(0)]*batch_size))#27*batch_size=10
    cost=criterion(preds,text,preds_size,length)/batch_size

    cost.backward()
    optimizer.step()
    return cost

#--------------------------------------------------------
def val(net,criterion):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad=False

    net.eval()
    val_iter=iter(val_loader)

    n_correct=0
    loss_avg=utils.averager()

    max_iter=len(val_loader)
    for i in range(max_iter):
        data=val_iter.next()
        cpu_images,cpu_texts=data #图像img 和labels str
        batch_size=cpu_images.size(0)
        utils.loadData(image,cpu_images)
        t,l=converter.encode(cpu_texts)
        utils.loadData(text,t)
        utils.loadData(length,l)

        preds=crnn(image)#[26,batch_size,37]
        preds_size=Variable(torch.LongTensor([preds.size(0)]*batch_size))#[26*batch_size]
        cost=criterion(preds,text,preds_size,length)/batch_size
        loss_avg.add(cost)

        #对应的indices
        """
        preds=torch.Tensor([[[7,3,1,4,2,6],
                     [1,2,6,3,5,4],
                     [5,6,4,8,7,1]]])
        preds.size(),preds.max(2)
        (torch.Size([1, 3, 6]),
        torch.return_types.max(
        values=tensor([[7., 6., 8.]]),
        indices=tensor([[0, 2, 3]])))
        _,preds=preds.max(2)
        preds
        tensor([[0, 2, 3]])"""
        #[26,10,37],最后一维概率最大的indices
        _,preds=preds.max(2)
        #[batch_size,26]有些tensor并不是占用一整块内存，而是由不同的数据块组成，
        #而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
        preds=preds.transpose(1,0).contiguous().view(-1)
        #拉成一行每个的数字预测码：[batch_size1,batch_size2,bacth_size3.....]
        sim_preds=converter.decode(preds.data,preds_size.data,raw=False)
        #str: [batch_size1,batch_size2,bacth_size3.....]

        for pred,target in zip(sim_preds,cpu_texts):
            if pred==target.lower():
                n_correct+=1

        raw_preds=converter.decode(preds.data,preds_size.data,raw=True)[:params.n_val_disp]
        #未加工的只取得前面10个字符
        for raw_pred,pred,gt in zip(raw_preds,sim_preds,cpu_texts):
            print('%-20s ——> %-20s,gt: %-20s '% (raw_pred,pred,gt))

    accuracy=n_correct/float(max_iter*params.batchSize)
    print('Val loss: %f,accuravy: %f' % (loss_avg.val(),accuracy))

#---------------------------------------------------------------------------
if __name__=="__main__":
    for epoch in range(params.nepoch):
        train_iter=iter(train_loader)
        i=0
        while i<len(train_loader):
            cost=train(crnn,criterion,optimizer,train_iter)
            loss_avg.add(cost)
            i+=1

            if i %params.displayInterval==0:#100
                print('[%d/%d][%d/%d] Loss: %f' % (epoch,params.nepoch,i,len(train_loader),loss_avg.val()))
                loss_avg.reset()#100个trainloader计算一次avg

            if i % 400==0:#1000
                val(crnn,criterion)

            #do checkpoint
            if i % 400==0:#1000
                torch.save(crnn.state_dict(),'{0}/netCRNN_{1}_{2}.pth'.format(params.expr_dir,epoch,i))
                #../CRNN