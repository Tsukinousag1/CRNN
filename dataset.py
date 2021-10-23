import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np

class lmdbDataset(Dataset):
    def __init__(self,root=None,transform=None,target_transform=None):
        self.env=lmdb.open(root,
                           max_readers=1,
                           #只读 read()
                           readonly=True,
                           lock=False,
                           readahead=False,
                           meminit=False)
        if not self.env:
            #如果没有此root下的数据库
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            #读图片操作
            nSamples=int(txn.get('num_Samples'.encode()))#S是大写的
            #将数据库中num_samples的value值处理为int
            self.nSamples=nSamples

        self.transform=transform
        self.target_transform=target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index<=len(self),'index range error'
        index+=1
        #下标从1开始
        with self.env.begin(write=False) as txn:
            img_key='image-%09d'%index
            #key:image-1
            #得到图像二进制value值
            imgbuf=txn.get(img_key.encode('utf-8'))
            #StringIO操作的只能是str，如果要操作二进制数据，就需要使用BytesIO。
            #BytesIO实现了在内存中读写bytes，我们创建一个BytesIO，然后写入一些bytes
            buf=six.BytesIO()
            #往内存中写入imgbuf，图像二进制数据
            buf.write(imgbuf)
            #File.seek(1) File.seek(2)
            #0指针回到文件开头 1当前位置 2文件结尾
            #对一个空文件写后再读时候，应在写完之后seek(0),使指针回到文件开头以便再读
            buf.seek(0)

            try:
                #转为灰度图像
                img=Image.open(buf).convert('L')
            except IOError:
                #损坏的图像，查看下一副
                print('Corrupted image for %d' % index)
                return self[index+1]

            if self.transform is not None:
                img=self.transform(img)

            label_key='label-%09d' % index
            #key:label-1
            label=str(txn.get(label_key.encode('utf-8')))
            #########对label进行去掉b‘’的操作，b''
            len_label = len(label) - 1
            label = label[2:len_label]
            label=label.lower()
            #########

            if self.target_transform is not None:
                label=self.target_transform(label)

        return (img,label)#图像img和label str
#由image-id和label-id查询lmdb中的图片数据和字符数据，得到一对训练样本：一张图片对应一个字符串

class resizeNormalize(object):
    #双线性插值是插值方法的一种，对图像进行resize；
    def __init__(self,size,interpolation=Image.BILINEAR):
        self.size=size
        self.interpolation=interpolation
        self.toTensor=transforms.ToTensor()
        #transforms.ToTensor()函数的作用是将原始的PILImage格式或者numpy.array格式的数据格式
        #化为可被pytorch快速处理的张量类型
        #ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可

    def __call__(self,img):
        #将img调整大小为(size,interpolation)
        img=img.resize(self.size,self.interpolation)
        #img转为tensor格式
        #先将输入归一化到(0, 1)
        img=self.toTensor(img)
        #下面相当于使用transforms.Normalize()，transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))
        #将每个元素分布到(-1,1)
        img.sub_(0.5).div_(0.5)
        return img

#随机顺序采样
class randomSequentialSampler(sampler.Sampler):

    def __init__(self,data_source,batch_size):
        #样本数
        self.num_samples=len(data_source)
        #批
        self.batch_size=batch_size

    def __iter__(self):
        #n批
        n_batch=len(self) // self.batch_size
        tail=len(self) % self.batch_size

        #初始化index
        index=torch.LongTensor(len(self)).fill_(0)
        #index=torch.LongTensor(6).fill_(0)
        #index  tensor([0, 0, 0, 0, 0, 0])

        for i in range(n_batch):
            #对于batch[0,1,2,3]
            random_start=random.randint(0,len(self)-self.batch_size)
            #[0,len(self)-batch_size]里面随机选择一个起点
            batch_index=random_start+torch.range(0,self.batch_size-1)
            #假设批是5个数
            #batch_index为[0，1，2，3，4]+随机起点
            index[i*self.batch_size:(i+1)*self.batch_size]=batch_index
            #[0,5][5,10]....其中的5个为上面的随机顺序采样

            #deal with tail
        if tail:
            random_start=random.randint(0,len(self)-self.batch_size)
            tail_index=random_start+torch.range(0,tail-1)
            #tail=3 [0,1,2]
            index[(i+1)*self.batch_size:]=tail_index

        return iter(index)
        #index为随机顺序采样batch_size的拼接


    def __len__(self):
        return self.num_samples

class alignCollate(object):
    def __init__(self,imgH=32,imgW=100,keep_ratio=False,min_ratio=1):
        """keep_ratio解析。假设原始图像大小为（1500， 1000），ratio=长边/短边 = 1.5。
            当keep_ratio=True时
            img_scale的多尺度最多为两个。假设多尺度为[(2000, 1200), (1333, 800)]
            首先将图像的短边固定到800到1200范围中的某一个数值假设为1100，
            那么对应的长边应该是短边的ratio=1.5倍
            且长边的取值在1333到2000的范围之内。如果大于2000按照2000计算，小于1300按照1300计算。
            当keep_ratio=False时
            img_scale的多尺度可以为任意多个。假设多尺度为[(2000, 1200), (1666, 1000),(1333, 800)]，则代表的含义为：随机从三个尺度中选取一个作为图像的尺寸进行训练。"""
        self.imgH=imgH
        self.imgW=imgW
        self.keep_ratio=keep_ratio
        self.min_ratio=min_ratio

    def __call__(self, batch):
        images,labels=zip(*batch)
        #第一维是imgaes
        #第二维是labels
        imgH=self.imgH
        imgW=self.imgW

        if self.keep_ratio:
            #要调整的话
            ratios=[]
            for image in images:
                w,h=image.size
                ratios.append(w/float(h))
                #计算调整率
            ratios.sort()
            max_ratio=ratios[-1]
            imgW=int(np.floor(max_ratio*imgH))#向下取整
            #最小比例是1
            imgW=max(imgH*self.min_ratio,imgW)
        #双线性插值
        transform=resizeNormalize((imgW,imgH))
        images=[transform(image) for image in images]
        #增加一个维度，在0维度上cat
        images=torch.cat([t.unsqueeze(0) for t in images],0)
        """y=torch.randn(2,3)
        y.shape
        torch.size([2,3])
        c=y.unsqueeze(0)
        c.shape
        torch.size([1,2,3])"""

        return images,labels
        #images:[imgW,imgH]
