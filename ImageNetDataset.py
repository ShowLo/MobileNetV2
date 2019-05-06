# -*- coding: UTF-8 -*-

'''
Ref: https://github.com/miraclewkf/MobileNetV2-PyTorch/blob/master/read_ImageNetData.py
'''

from torchvision import transforms, datasets
import os
import torch
from PIL import Image
import scipy.io as scio

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def ImageNetDataset(args):
    # data transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}
    #image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'ILSVRC2012_img_train'), data_transforms['train'])

    image_datasets['train'] = ImageNetTrainDataset(os.path.join(args.data_dir, 'ILSVRC2012_img_train'),
                                           os.path.join(args.data_dir, 'ILSVRC2012_devkit_t12', 'data', 'meta.mat'),
                                           data_transforms['train'],
                                           num_class=args.num_class)
    image_datasets['val'] = ImageNetValDataset(os.path.join(args.data_dir, 'ILSVRC2012_img_val'),
                                               os.path.join(args.data_dir, 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt'),
                                               data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes

class ImageNetTrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_label, data_transforms, num_class):
        # 训练图片及其标签的对应关系(the map between train file name and label)
        label_array = scio.loadmat(img_label)['synsets']
        label_dic = {}
        # 将字符串形式的标签转为数字形式的编号(Converting String-Form Labels to Numbers in Digital Form)
        for i in range(num_class):
            label_dic[label_array[i][0][1][0]] = i
        self.data_transforms = data_transforms
        self.label_dic = label_dic
        self.root_dir = root_dir
        self.imgs = self._make_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        img = Image.open(path).convert('RGB')
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(path))
        return img, label

    def _make_dataset(self):
        class_to_idx = self.label_dic
        images = []
        dir = os.path.expanduser(self.root_dir)
        for target in sorted(os.listdir(dir)):
            # 存放各个类图片的文件夹(the folders for storing various types of pictures)
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self._is_image_file(fname):
                        path = os.path.join(root, fname)
                        # 图片路径及其对应的标签(Image path and its corresponding label)
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images

    def _is_image_file(self, filename):
        '''
        检查文件是否为图片
        Checks if a file is an image.
        '''
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, img_label_path, data_transforms):
        self.data_transforms = data_transforms
        img_names = os.listdir(img_path)
        img_names.sort()
        self.img_path = [os.path.join(img_path, img_name) for img_name in img_names]
        with open(img_label_path, 'r') as input_file:
            lines = input_file.readlines()
            self.img_label = [(int(line)-1) for line in lines]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx]).convert('RGB')
        label = self.img_label[idx]
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[idx]))
        return img, label