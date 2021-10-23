import torch
import torch.nn as nn
from torch.autograd import Variable
import collections


class strLabelConverter(object):
    #convert between str and lebel

    #NOTE:
    #   Insert 'blank' to the alphabet for CTC
    #Args:
    #   alphabet(str):set of possible characters
    #是否忽视大小写
    #ignore_cass(bool,default=True):whether or not to ignore all of the case
    def __init__(self,alphabet,ignore_case=True):
        #aplhabet用来确定输出的类别中每个类别对应的字符，是一个字符串形式
        self._ignore_case=ignore_case
        if self._ignore_case:
            alphabet=alphabet.lower()
            #全部大写字母转为小写字母
        self.aplhabet=alphabet+'-'
        #对于 ‘-1’ index

        self.dict={}
        #构建映射表
        for idx,char in enumerate(alphabet):
            #NOTE:0 is reserved for 'blank' required by wrap_ctc
            #由ctc要求，index 0 第一位为‘blank’，因此从1开始映射
            #字符 ——→ idx
            self.dict[char]=idx+1

    def encode(self,text):
        """Support batch or single str

        :param
            text(str or list of str):text to convert.
        :return:
            torch.IntTensor[idx_0,idx_1,...+idx_{n-1}]:encoded texts
            torch.IntTensor[n]:length of each text.
        """
        length=[]
        result=[]
        for item in text:

            length.append(len(item))
            r=[]
            for char in item:
            #b'deltas'
                index=self.dict[char]
                r.append(index)
            result.append(r)
            #result:[[0,1,3,6,1,0],[0,1,2,3,4,6],[0,1,2,3]]
        max_len=0
        for r in result:
            if len(r)>max_len:
                max_len=len(r)
        #取最大长度进行统一，长度没统一的后面用0补齐
        result_temp=[]
        for r in result:
            for i in range(max_len-len(r)):
                r.append(0)
            result_temp.append(r)

        text=result_temp
        return (torch.IntTensor(text),torch.IntTensor(length))
#a=[1,2,3,4,5]
#text=torch.IntTensor(a)
#text  tensor([1, 2, 3, 4, 5], dtype=torch.int32)
#length tensor([5],dtype=torch.int32)

    def decode(self,t,length,raw=False):
        """
        Decode encode texts back into strs.

        :param t:
            torch.IntTensor text[1,2,3,4,5,6]
        :param length:
            torch.IntTensor length
        :param raw:
            when the texts ans its length does not match.
        :return:
            text(str or list of str):texts to convert.
        """
        if length.numel()==1:
            length=length[0]
            assert t.numel()==length,"text with length:{} does not match declared length:{}".format(t.numel(),length)
            if raw:#未加工过的a__a__b__d——→aabd
                #由于t是从1开始映射的
                print(t)
                return ''.join([self.aplhabet[i-1] for i in t])
            else:#加工过的,进行去重操作a__a__bb__cd——→aabcd
                char_list=[]
                for i in range(length):
                    if t[i]!=0 and (not(i>0 and t[i-1]==t[i])):#i=0或者a[i]!=a[i-1]满足
                        char_list.append(self.aplhabet[t[i]-1])
                return ''.join(char_list)
        else:
            #batch mode
            assert t.numel()==length.sum(),"texts with length：{} does not match declared length:{}".format(t.numel(),length.sum())
#a=[[1,2,3,4],[1,2,6,2]]
#b=[4,4]
#text=torch.IntTensor(a)
#length=torch.IntTensor(b)
#text.numel()==length.sum()
#此处默认，每一次的text的长度必须统一
            texts = []
            index = 0
            for i in range(length.numel()):
                l=length[i]
                #以单字符串的形式，加入list texts
#print(text[0:1])
#tensor([[1, 2, 3, 4]], dtype=torch.int32)
                texts.append(
                    self.decode(
                        t[index:index+l],torch.IntTensor([l]),raw=raw))
                index+=l
            return texts
            #结果字符串


class averager(object):
    """
        Compute average for torch.Variable and torch.Tensor
    """
    def __init__(self):
        self.reset()

    def add(self,v):
        if isinstance(v,Variable):
#x = Variable(torch.Tensor([2, 5]), requires_grad=True)
#x.data, x.data.numel(), x.data.sum()
#(tensor([2., 5.]), 2,  tensor(7.))
#y=torch.Tensor([1,2,3])
#y.numel()
#3
            count=v.data.numel()#2
            v=v.data.sum()
        elif isinstance(v,torch.Tensor):
            count=v.numel()
            v=v.sum()

        self.n_count+=count
        self.sum+=v

    def reset(self):
        self.n_count=0
        self.sum=0

    def val(self):
        res=0
        if self.n_count!=0:
            res=self.sum/float(self.n_count)
            #tensor([2., 5.],平均3.5
        return res

################################################################
def oneHot(v,v_length,nc):
    batchSize=v_length.size(0)
    maxLength=v_length.max()
    v_onehot=torch.FloatTensor(batchSize,maxLength,nc).fill_(0)
    #初始化size为[batchsize,maxlength,nc]全 0 tensor
    acc=0
    for i in range(batchSize):
        length=v_length[i]
        label=v[acc:acc+length].view(-1,1).long()
        #label是hello:[][][][][],[]里面都是二进制数，拉成1列
        #.long()转为torch.int64
        #第一个维度按列填充,以laebl的形状,填1
        v_onehot[i,:length].scatter_(1,label,1.0)
        #1是在行里面填充
        acc+=length
    return v_onehot
##################################################################

def loadData(v,data):
    v.resize_(data.size()).copy_(data)
    #此处必须用resize_改变Tensor大小不可以用reshape_
    #会修改a的shape

def prettyPrint(v):
    print('Size{0},Type:{1}'.format(str(v.size()),v.data.type()))
    print('|Max: %f|Min: %f| Mean: %f' % (v.max().data[0],
                                          v.min().data[0],
                                          v.mean().data[0]))
####################################################################
def assureRatio(img):
    """Ensure imgH<=imgW"""
    b,c,h,w=img.size()
    #采用上采样,平均法
    if h>w:
        main=nn.UpsamplingBilinear2d(size=(h,h),scale_factor=None)
        img=main(img)
    return img

#注：loaddata preetyprint oneHot有些看不懂