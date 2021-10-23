#about data and net
alphabet='0123456789abcdefghijklmnopqrstuvwxyz'
keep_ratio=False    #wether to keep ratio for image resize
manualSeed=1234     #reproduce experiemnt
random_sample=True  #whether to sample the dataset with random sample 是否随即顺序采样
imgH=32
imgW=100
nh=256              #size of lstm hidden state
nc=1                #输入通道数
pretrained=''       #path to pretrained model(to continue traning)
expr_dir='/home/std2021/hejiabang/OCR/CRNN'     #where to store samples and models 保存参数和模型
dealwith_lossnan=False #whether to replace all nan/inf in gradients to zero

#hardware
cuda=False   #enables cuda
multi_gpu=False
ngpu=0
workers=0

#training process
displayInterval=100     #interval to be print the train loss 间隔打印train loss
valinterval=1000        #interval to val the model loss and accuracy
saveInterval=1000       #interval to save model
n_val_disp=10           #number of samples to display when val the model

#finetune
nepoch=1000             #number of epochs to train for
batchSize=64            #input batch size
lr=0.0001               #learning rate for Critic,not used by adadealta
beta1=0.5               #beta1 for adam. default=0.5

adam=False              #wether to use adam(default is rmsprop)
adadelta=False          #wether to use adadelts(default is rmsprop)

