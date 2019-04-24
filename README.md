# 基于pytorch的ILSVRC2012训练

在读论文的时候时常看到使用基于imagenet-1k的预训练模型，所以写了这个rep.常用的网络基本都可以从网上下载的到，但有一些修改过的基础框架可能没法从网络上下载到预训练模型。
### 数据准备
把valprep.sh 复制到val的图像文件夹中，然后在命令行中运行
```
sh valpre.sh
```
valpre.sh脚本文件会按照训练集的文件结构把对应的图片放到相应的文件夹中。

### 数据生成器
```
from  imagenet  import ILSVRC 
train_dataset = ILSVRC(ilsvrc_data_path='data',meta_path='data/meta.mat')
val_dataset =  ILSVRC(ilsvrc_data_path='data',meta_path='data/meta.mat',val=True)
```
上述代码中的meta_path参数设置为验证集中的meta.mat的路径。
### 需要的包

* pytorch==1.0.0
* python==3.6
* numpy
* torchvision
* matplotlib
* opencv-python
* tensorflow
* tensorboardX

### 训练

* 在终端中运行
```
python main.py
```
* 每个epoch会在验证集上评测一次。评价指标目前只采用了top1 acc。修改评测频数可以在trainer.py中修改。一些超参数可以在main.py中设置。
* 模型的断点续训练可以去除main.py如下代码的#号。这样就会接着上一次中断的epoch继续训练。
```
    trainer = Trainer('training', optimizer,exp_lr_scheduler, net, cfg, './log')
    #trainer.load_weights(trainer.find_last())
    trainer.train(train_loader, val_loader, criterion, 60)
```

