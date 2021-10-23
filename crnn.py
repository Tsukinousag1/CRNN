import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    """
    net = BidirectionalLSTM(256, 256, 10)  # In,Hidden,Out
    x = torch.randn(16, 16, 256)  # 输入维度[16,16,256]
    out = net(x)
    print(out.shape)
    [16,16,256]->[16,16,10]
    """
    def __init__(self,nIn,nHidden,nOut):#[256,256,10]
        super(BidirectionalLSTM,self).__init__()

        self.rnn=nn.LSTM(nIn,nHidden,bidirectional=True)
        self.embedding=nn.Linear(nHidden*2,nOut)#[256*2,10]

    def forward(self,input):
        recurrent,_=self.rnn(input)
        #recurrent.shape:[16,16,512],正向+反向
        T,b,h=recurrent.size()
        #16 16 512
        t_rec=recurrent.view(T*b,h)
        #[256,512]->[长*宽，通道数]
        output=self.embedding(t_rec)
        #[256,10],embedding到最后一个维度是10
        output=output.view(T,b,-1)
        #[16,16,10]
        return output
class CRNN(nn.Module):
    def __init__(self,imgh,nc,nclass,nh,n_rnn=2,leakyRelu=False):
        super(CRNN,self).__init__()
        assert imgh%16==0,"imgh has to be a multiple of 16"
        #判断是否是16的倍数
        ks=[3,3,3,3,3,3,2]#kernel size
        ps=[1,1,1,1,1,1,0]#paddding
        ss=[1,1,1,1,1,1,1]#stride
        nm=[64,128,256,256,512,512,512]#输出
        #网路配置

        cnn=nn.Sequential()

        def convRelu(i,batchNormalization=False):
            #conv+relu
            nIn=nc if i==0 else nm[i-1]
            #i=0时输入维度是nc，i=1时刻开始输入维度是nm[0],也就是64
            nOut=nm[i]
            #i=0时刻输出维度是64，i=1时刻输出维度是128
            cnn.add_module('conv{0}'.format(i),nn.Conv2d(nIn,nOut,ks[i],ss[i],ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i),nn.BatchNorm2d(nOut))

            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),nn.LeakyReLU(0.2,inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i),nn.ReLU(True))

        #torch.Size([10, 1, 32, 100])
        convRelu(0)#(1,64,(3,3),(1,1),(1,1))#[n,c,h,w]
        # torch.Size([10, 64, 32, 100])
        cnn.add_module('pooling{0}'.format(0),nn.MaxPool2d(2,2))
        #torch.Size([10, 64, 16, 50])
        convRelu(1)
        ##torch.Size([10, 128, 16, 50])
        cnn.add_module('pooling{0}'.format(1),nn.MaxPool2d(2,2))
        # torch.Size([10, 128, 8, 25])
        convRelu(2,True)
        #torch.Size([10, 256, 8, 25])
        convRelu(3)
        #让高度和宽度方向不同池化时,kernelsize,stride,padding
        #torch.Size([10, 256, 8, 25])
        cnn.add_module('pooling{0}'.format(2),nn.MaxPool2d((2,2),(2,1),(0,1)))
        #torch.Size([10, 256, 4, 26])
        convRelu(4,True)
        #torch.Size([10, 512, 4, 26])
        convRelu(5)
        #torch.Size([10, 512, 4, 26])
        cnn.add_module('pooling{0}'.format(3),nn.MaxPool2d((2,2),(2,1),(0,1)))
        #torch.Size([10, 512, 2, 27])
        convRelu(6,True)
        #torch.Size([10, 512, 1, 26])

        self.cnn=cnn
        self.rnn=nn.Sequential(
            BidirectionalLSTM(512,nh,nh),
            BidirectionalLSTM(nh,nh,nclass)
        )

    def forward(self,input):
        conv=self.cnn(input)
        #样本数，通道数，高，宽 n,c,h,w
        b,c,h,w=conv.size()
        #print(b,c,h,w)
        #10 512 1 26

        assert h==1,"the height of conv must be 1"

        conv=conv.squeeze(2)#去除h的维度
        conv=conv.permute(2,0,1)#[w,b,c]宽，样本数，通道数
        #torch.Size([26, 10, 512])

        #rnn features
        output=self.rnn(conv)
        #torch.Size([26, 10, 37])

        return output

#测试
"""img=torch.randn(10,1,32,100)#[n,c,h,w]
net=CRNN(32,1,37,256)
#确定输入图片高为32像素，输入的图片通道数是1，类别为37，RNN的隐藏层单元nh数目为256
output=net(img)
print(output.shape)"""
#torch.Size([26, 10, 37])
#宽，样本数，通道数
