import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import crnn as crnn
import params
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/home/std2021/hejiabang/OCR/CRNN/sp/netCRNN_900_449.pth', help='path to pth')  # ../CRNN
parser.add_argument('--image_path', type=str, default='/home/std2021/hejiabang/OCR/CRNN/demo_img.jpg', help='path to img')  # ../CRNN
args=parser.parse_args()

model_path=args.model_path
image_path=args.image_path

#net init
nclass=len(params.alphabet)+1
model=crnn.CRNN(params.imgH,params.nc,nclass,params.nh)
if torch.cuda.is_available():
    model=model.cuda()

#load model
print('loading pretrained model from %s ' % model_path)
if params.multi_gpu:
    model=torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path))

converter=utils.strLabelConverter(params.alphabet)

#将图片大小调整到100*32
transforms=dataset.resizeNormalize((100,32))

#打开image
image=Image.open(image_path).convert('L')

image=transforms(image)

if torch.cuda.is_available():
    image=image.cuda()

image=image.view(1,*image.size())#(1,100,32)
image=Variable(image)

model.eval()
preds=model(image)
#torch.Size([26, 10, 37])

_,preds=preds.max(2)
#_表示最大值，preds表示最大下标,[26,1]——>[1,26]
preds=preds.transpose(1,0).contiguous().view(-1)

preds_size=Variable(torch.LongTensor([preds.size(0)]))

raw_pred=converter.decode(preds.data,preds_size.data,raw=True)
sim_pred=converter.decode(preds.data,preds_size.data,raw=False)

print('%-20s => %-20' % (raw_pred,sim_pred))