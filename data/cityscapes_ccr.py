import os
import PIL
import numpy as np
import torch
from absl import flags

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

flags.DEFINE_string('cityscapes_dir', None, 'Cityscapes Data Directory')
flags.DEFINE_boolean('augmentations', True, 'Use augmentations while training')

flags.mark_flag_as_required('cityscapes_dir')
opts = flags.FLAGS

class TrainData(torch.utils.data.Dataset):
    def __init__(self, opts):
        super(TrainData, self).__init__()
        file_path = os.path.join(opts.cityscapes_dir, 'leftImg8bit', 'train')
        self.files = self.get_city_files(file_path)
        print('No. of Training CCR Images : ', len(self.files))
        self.image_height = opts.image_height
        self.image_width = opts.image_width
        self.augment = opts.augmentations
        if self.augment:
            import albumentations as A
            self.transform = A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2)
    
    @staticmethod
    def get_city_files(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    file_name = os.path.join(r, file)
                    files.append(file_name)
        
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_name = self.files[idx]
        I = PIL.Image.open(im_name)
        sI = I.resize((self.image_width, self.image_height), PIL.Image.ANTIALIAS)
        aI = np.array(sI, dtype=np.uint8)

        if self.augment:
            augmented = self.transform(image=aI)
            aI = augmented['image']

        hedI = cv2.cvtColor(aI.copy(), cv2.COLOR_RGB2BGR)
        
        sample = {'image': np.moveaxis(aI, 2, 0).astype('float') / 255.0, 'hed_image': np.moveaxis(hedI, 2, 0)}

        return sample



class ValData(torch.utils.data.Dataset):
    def __init__(self, opts):
        super(ValData, self).__init__()
        file_path = os.path.join(opts.cityscapes_dir, 'leftImg8bit', 'val')
        self.files = self.get_city_files(file_path)
        print('No. of Validation CCR Images : ', len(self.files))
        self.image_height = opts.val_image_height
        self.image_width = opts.val_image_width
        
    @staticmethod
    def get_city_files(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    file_name = os.path.join(r, file)
                    files.append(file_name)
        
        return files
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_name = self.files[idx]
        I = PIL.Image.open(im_name)
        sI = I.resize((self.image_width, self.image_height), PIL.Image.ANTIALIAS)
        aI = np.array(sI, dtype=np.uint8)
        hedI = cv2.cvtColor(aI.copy(), cv2.COLOR_RGB2BGR)
        
        sample = {'image': np.moveaxis(aI, 2, 0).astype('float') / 255.0, 'hed_image': np.moveaxis(hedI, 2, 0)}

        return sample

