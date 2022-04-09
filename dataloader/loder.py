import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

class Getdata(torch.utils.data.Dataset):
    def __init__(self, is_train, data_root,crop_size,pattern="random"):
        '''
        is_train: True或False
        data_root: 数据集主目录名
        crop_size:(896,896)
        pattern:random或者middle
        '''
        self.crop_size = crop_size
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)
        ])
        self.root = data_root
        self.pattern = pattern
        images,labels,self.classes = self.read_file_list(root=self.root,is_train=is_train)
        self.images,self.labels = self.filter(images,labels,crop_size)  # images list
        print('Read ' + str(len(self.images)) + ' valid examples')

    
    # 过滤掉尺寸小于crop_size的图片
    def filter(self, images,labels,crop_size): 
        image_set = []
        label_set = []
        for i in range(len(images)):
            if (Image.open(images[i]).size[1] >= crop_size[0] and
                Image.open(images[i]).size[0] >= crop_size[1]):
                image_set.append(images[i])
                label_set.append(labels[i])
        return image_set,label_set
    
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.open(image).convert('RGB')

        image = self.rand_crop(image,*self.crop_size,self.pattern)
        image = self.tsf(image)

        return image, self.labels[idx]  # float32 tensor, uint8 tensor

    def __len__(self):
        return len(self.images)
    
    def read_file_list(self,root, is_train=True):
        #函数读取所有输入图像和标签的文件路径。
        txt_fclass_name = os.path.join(root,'classes.txt')
        with open(txt_fclass_name, 'r') as f:
            class_set_filenames = f.read().split()
        one_hot = np.eye(len(class_set_filenames))
        classes2onehot = {class_name : o_hot for class_name,o_hot in zip(class_set_filenames,one_hot)}
        classes = {o_hot.argmax() : class_name for class_name,o_hot in zip(class_set_filenames,one_hot)}
        data_set_filenames = []
        txt_fdata_name = os.path.join(root,'train.txt' if is_train else 'val.txt')
        f = open(txt_fdata_name)
        while True:
            line = f.readline()
            if line:
                data_set_filenames.append(line)
            else:
                break
        f.close()
        labels = [classes2onehot[data_set_filenames[_][:data_set_filenames[_].rfind("\\")]] for _ in range(len(data_set_filenames))]
        images = [os.path.join(root, 'Images', i.replace("\n","")) for i in data_set_filenames]
        return images,labels,classes
    
    def rand_crop(self,image,height,width,pattern):
        """
        图片随机切割
        """
        if pattern == "random":
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(height, width))
            image = transforms.functional.crop(image, i, j, h, w)
        elif pattern == "middle":
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(height, width))
            image = transforms.functional.crop(image, i, j, h, w)
            centerCrop = transforms.CenterCrop(size=(height//2, width//2))
            image = centerCrop(image)
        
        resize = transforms.Resize([height//2,width//2])
        return resize(image)

    def show(self,key):
        """
        图片展示
        """
        image = Image.open(self.images[key]).convert('RGB')
        image = self.rand_crop(image,*self.crop_size,self.pattern)
        label = self.classes[self.labels[key].argmax()]
        print(label)
        plt.imshow(image)
        plt.show()