from  imagenet  import ILSVRC 
import argparse
from torch.utils.data import  DataLoader
from pathlib import Path
import yaml
from trainer import Trainer
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from model.backbone import xceptionA
from config import Config


if __name__=='__main__':

    cfg=Config()
    #create dataset
    train_dataset = ILSVRC(ilsvrc_data_path='data',meta_path='data/meta.mat')
    val_dataset =  ILSVRC(ilsvrc_data_path='data',meta_path='data/meta.mat',val=True)

    train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=128, shuffle=True,
                                           num_workers=8)
    val_loader = DataLoader(dataset=val_dataset,
                                         batch_size=128, shuffle=False, 
                                         num_workers=8)
    net = xceptionA(num_classes=1000)
    #load loss
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(
    net.parameters(), lr=0.3, momentum=0.9,weight_decay=0.00004)  #select the optimizer

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,30,gamma=0.1)
    trainer = Trainer('training', optimizer,exp_lr_scheduler, net, cfg, './log')
    trainer.load_weights(trainer.find_last())
    trainer.train(train_loader, val_loader, criterion, 60)
    #trainer.evaluate(valid_loader)
    print('Finished Training')
