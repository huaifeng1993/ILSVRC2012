import torch
import torch.utils.data
import numpy as np 
import cv2
import os
from scipy.io import loadmat
from PIL import Image
import random

class ILSVRC(torch.utils.data.Dataset):
    def __init__(self,ilsvrc_data_path,meta_path,val=False):
        
        if val:
            synset_dir=os.path.join(ilsvrc_data_path,'val')
        else:
            synset_dir=os.path.join(ilsvrc_data_path,'train')
        
        wnids=self.load_imagenet_meta(meta_path)[0][:1000]
        images_dirs=[os.path.join(synset_dir,str(wnid)) for wnid in wnids]
        self.examples=[]
        print('loading images ...')
        for i,dir in enumerate(images_dirs):
            images_id=os.listdir(dir)
            for id in images_id:
                example={}
                example['image_path']=os.path.join(dir,id)
                example['label']=i
                self.examples.append(example)
       
    def __getitem__(self,index):
        example = self.examples[index]
        img=self.preprocess_image(example['image_path'])
        img=np.transpose(img,(2,0,1))
        label=example['label']

        img=torch.from_numpy(img)
        return (img,label)

    def __len__(self):
        return len(self.examples)

    def preprocess_image(self,image_path):
        """ It reads an image, it resize it to have the lowest dimesnion of 256px,
            it randomly choose a 224x224 crop inside the resized image and normilize the numpy 
            array subtracting the ImageNet training set mean
            Args:
                images_path: path of the image
            Returns:
                cropped_im_array: the numpy array of the image normalized [width, height, channels]
        """
        IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format

        img = Image.open(image_path).convert('RGB')

        # resize of the image (setting lowest dimension to 256px)
        if img.size[0] < img.size[1]:
            h = int(float(256 * img.size[1]) / img.size[0])
            img = img.resize((256, h), Image.ANTIALIAS)
        else:
            w = int(float(256 * img.size[0]) / img.size[1])
            img = img.resize((w, 256), Image.ANTIALIAS)

        # random 244x224 patch
        x = random.randint(0, img.size[0] - 224)
        y = random.randint(0, img.size[1] - 224)
        img_cropped = img.crop((x, y, x + 224, y + 224))
        
        # data augmentation: flip left right
        if random.randint(0,1) == 1:
            img_cropped = img_cropped.transpose(Image.FLIP_LEFT_RIGHT)

        cropped_im_array = np.array(img_cropped, dtype=np.float32)

        for i in range(3):
            cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]

        return cropped_im_array/225



    def load_imagenet_meta(self,meta_path):
        """ It reads ImageNet metadata from ILSVRC 2012 dev tool file
            Args:
                meta_path: path to ImageNet metadata file
            Returns:
                wnids: list of ImageNet wnids labels (as strings)
                words: list of words (as strings) referring to wnids labels and describing the classes 
        """
        metadata = loadmat(meta_path, struct_as_record=False)

        # ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
        synsets = np.squeeze(metadata['synsets'])
        ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
        wnids = np.squeeze(np.array([s.WNID for s in synsets]))
        words = np.squeeze(np.array([s.words for s in synsets]))
        return wnids, words


if __name__=='__main__':
    train_dataset = ILSVRC(ilsvrc_data_path='data',meta_path='data/meta.mat')
    from torch.utils.data import  DataLoader

    train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=64, shuffle=True,
                                           num_workers=8)

    img,label=train_dataset[1]
    print(img.shape,label)
