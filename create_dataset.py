import argparse
import os
import lmdb
import cv2
import numpy as np

def checkImageIsValid(imageBin):
    #imageBin是字符串的形式图片
    if imageBin is None:
        return False
    #str转为np数组格式
    imageBin=np.fromstring(imageBin,dtype=np.uint8)
    img=cv2.imdecode(imageBin,cv2.IMREAD_GRAYSCALE)
    imgH,imgW=img.shape[0],img.shape[1]
    if imgH * imgW==0:
        return False
    return True

def writeCache(env,cache):
    with env.begin(write=True) as txn:
        for k,v in cache.items():
            if type(k)==str:
                k=k.encode()
            if type(v)==str:
                v=v.encode()
            txn.put(k,v)

def createDataset(outputPath,imagePathList,labelList,lexiconList=None,checkValid=True):
    """
    Create LMDB dataset for CRNN traning

    ARGS:
          outputPath    :LMDB output path
          imagePathList :list of image path
          labelList     :list of corresponding groundtruth tests
          lexiconList   :(optional) list of lexicon lists//可选词表列表
          checkValid    :if true,check the validity of every image
    """
    assert(len(imagePathList)==len(labelList))
    nSamples=len(imagePathList)
    env=lmdb.open(outputPath,map_size=int(1e9))
    cache={}
    #从1开始映射
    cnt=1
    for i in range(nSamples):
        imagePath=imagePathList[i]
        label=labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        #iamgeBin所在路径
        with open(imagePath,'rb') as f:########此处必须用rb，不可以用r——→此处出来的是该图片bytes形式
            imageBin=f.read()
            #read出的是bytes的形式

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey='image-%09d'%cnt
        labelKey='label-%09d'%cnt

        cache[imageKey]=imageBin#"bytes 形式得图像数据"
        cache[labelKey]=label#string 标签

        if lexiconList:
            lexiconKey='lexicon-%09d' % cnt
            cache[lexiconKey]=' '.join(lexiconList[i])

        if cnt%100==0:
            writeCache(env,cache)
            cache={}
            print('Written %d / %d' % (cnt,nSamples))

        cnt+=1

    nSamples=cnt-1
    cache['num_Samples']=str(nSamples)
    #最后映射存数量
    writeCache(env,cache)
    print('Create dataset with %d samples' % nSamples)

def read_data_from_folder(folder_path):
    image_path_list=[]
    label_file=[]
    label_list=[]
    pics=os.listdir(folder_path)
    for pic in pics:
        image_path_list.append(folder_path+'/'+pic)#图片路径就绪，下面是找对应的图片路径的label标签

        a = pic.split('.')[0]
        label_id = 'gt_' + a + '.txt'
        line =''
        label_file.append('gt_'+a+'.txt')
        dir = '../data/ICDAR2013/Challenge2_Training_Task1_GT/' + label_id
        f = open(dir)
        while True:
            line1 = f.readline()
            line1 = line1.replace('\n', '').replace('\r', '').replace(' ', '')
            tmp = ""
            flag=0
            for t in line1:
                if t >= '0' and t <= '9':
                    if flag == 0:
                        continue
                    else:
                        tmp += t
                elif t == '\"':
                    flag = 1
                    continue
                else:
                    tmp += t
            line += tmp
            if not line1:
                break
            line += ' '
        label_list.append(line.rstrip())
    return image_path_list,label_list


def read_data_from_folder2(folder_path):
    image_path_list=[]
    label_list=[]
    #pics存储folder_path路径下的所有图片名称
    pics=os.listdir(folder_path)
    pics.sort(key=lambda i:len(i))
    #sort只能对列表排序
    #key=lamda i 表明排序准则 -len(x)长度从大到小排序，len(x)长度从小到大排序
    # sort the image according to the text length.
    for pic in pics:
        image_path_list.append(folder_path+'/'+pic)
        label_list.append(pic.split('_')[0])
        # image_path_list=['../data/***/hi_o.jpg',....]
        # label_list=['hi',...]
        # pic='hi_0.jpg'
        # print(pic.split('_'))
        # ['hi', '0']
        # 'hi'
    return image_path_list,label_list

def read_data_from_file2(file_path):
    image_path_list=[]
    label_list=[]
    f=open(file_path)
    #   f = open('../data/hjb.txt')
    #   absolute / path / to / image / 一身转战_0.jpg
    #   一身转战
    #   absolute / path / to / image / 三千里_1.jpg
    #   三千里
    #   absolute / path / to / image / 一剑曾当百万师_2.jpg
    #   一剑曾当百万师
    while True:
        line1=f.readline()
        #readline()：一次读取一行数据。需要注意的是，每次读取出来的数据末尾都会有\n
        line2=f.readline()
        if not line1 or not line2:
            break
        line1=line1.replace('\r','').replace('\n','')
        line2=line2.replace('\r','').replace('\n','')
        image_path_list.append(line1)
        label_list.append(line2)
    return image_path_list,label_list

def read_data_from_file(file_path):
    #file_path = '../data/crnn_img/gt_img.txt'
    road = '/mnt/disk2/std2021/hejiabang-data/OCR/crnn_img/'
    image_path_list = []
    label_list = []
    f = open(file_path)
    while True:
        line1 = f.readline()
        line1.replace('\n', '').replace('\r', '')
        idx = 0
        tmp = ''
        t = 0
        for i in line1:
            if i == ' ':
                t = idx
            idx += 1
        idx = 0
        for i in line1:
            if idx == 6:
                tmp = line1[idx:t]
                slabel = ''
                for lab in tmp:
                    if lab >= 'a' and lab <= 'z' or lab >= 'A' and lab <= 'Z':
                        slabel += lab
                    if lab == '.':
                        break
                label_list.append(slabel)
                image_path_list.append(road +line1[2]+'/'+ line1[4] + '/' + tmp)
            idx += 1
        if not line1:
            break
    return image_path_list,label_list

def show_demo(demo_number,image_path_list,label_list):
    print("\nShow some demo to prevent creating wrong lmdb data")
    print('The first line is the path to image and the second line is the image label')
    for i in range(demo_number):
        print('image:%s\nlabel:%s\n' % (image_path_list[i],label_list[i]))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=str,default='/home/std2021/hejiabang/OCR/CRNN/db',help='lmdb data output path')# ../CRNN
    parser.add_argument('--folder',type=str,default='',help='path to folder which contains the images')# ../CRNN/gt_img #default='../data/ICDAR2013/Challenge2_Training_Task12_Images'
    parser.add_argument('--file',type=str,default='/mnt/disk2/std2021/hejiabang-data/OCR/crnn_img/gt_img.txt',help='path to file which contains the image path and label')#
    args=parser.parse_args()

    if args.file is not None:
        image_path_list,label_list=read_data_from_file(args.file)
        createDataset(args.out,image_path_list,label_list)
        show_demo(2,image_path_list,label_list)
    elif args.folder is not None:
        image_path_list,label_list=read_data_from_folder(args.folder)
        createDataset(args.out,image_path_list,label_list)
        show_demo(2,image_path_list,label_list)
    else:
        print('Please use --folder or --file to assign the input. Use -h to see more.')


#------------------------------------------------------------------
"""env=lmdb.open("../CRNN/db")

txn=env.begin(write=False)


t=txn.get('num_Samples'.encode())

t=str(t)

print(t,t[0])
#b'28711' b

txn.commit()

env.close()
"""
"""str='1230 1234 12134 13213 "Info@qq.com.de.8"'
line=''
line1 = str.replace('\n', '').replace('\r', '').replace(' ', '')
print(line1)
tmp = ""
flag=0
for t in line1:
    if t>='0' and t<='9':
        if flag==0:
            continue
        else :
            tmp+=t
    elif t=='\"':
        flag=1
        continue
    else:
        tmp+=t

line += tmp
print(line)"""
#----------------------------------------------------

"""file_path='../data/crnn_img/gt_img.txt'
road='../data/crnn_img/'
image_path_list=[]
label_list=[]
f=open(file_path)

while True:
    line1=f.readline()
    line1.replace('\n','').replace('\r','')
    idx=0
    tmp=''
    t=0
    for i in line1:
        if i==' ':
            t=idx
        idx+=1
    idx=0
    for i in line1:
        if idx==6:
            tmp=line1[idx:t]
            slabel=''
            for lab in tmp:
                if lab>='a' and lab<='z' or lab>='A' and lab<='Z':
                    slabel+=lab
                if lab=='.':
                    break
            label_list.append(slabel)
            image_path_list.append(road+line1[4]+'/'+tmp)
        idx+=1
    if not line1:
        break"""




