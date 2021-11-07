import os
import PIL
from albumentations.augmentations.transforms import Rotate
import numpy as np
import torch
from absl import flags

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

flags.DEFINE_string('cityscapes_dir', None, 'Cityscapes Data Directory')
flags.DEFINE_boolean('augmentations', True, 'Use augmentations while training')

flags.DEFINE_integer('max_depth_dataset', 32000, 'Maximum Depth in the dataset, values above will be clipped')
flags.DEFINE_float('max_depth', 32.0, 'Maximum Depth of the output, values above will be clipped')
flags.DEFINE_float('depth_threshold', 1.0, 'Threshold of Depth in the dataset, values below wont count in the loss')

flags.mark_flag_as_required('cityscapes_dir')
opts = flags.FLAGS

class TrainData(torch.utils.data.Dataset):
    def __init__(self, opts, sigma=8):
        super(TrainData, self).__init__()
        file_path = os.path.join(opts.cityscapes_dir, 'leftImg8bit', 'train')
        self.files = self.get_city_files(file_path)
        print('No. of Training Images : ', len(self.files))
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.inst_height = opts.image_height
        self.inst_width = opts.image_width
        self.max_depth = opts.max_depth_dataset
        self.max_out = opts.max_depth
        self.depth_threshold = opts.depth_threshold
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
    
    @staticmethod
    def replace_ids(n_arr):
        arr = 19 * np.ones(n_arr.shape, dtype=n_arr.dtype)

        ids = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
            25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for i in ids.keys():
            if i in ids:
                arr[n_arr == i] = ids[i]

        return arr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_name = self.files[idx]
        sg_name = self.files[idx].replace('leftImg8bit', 'gtFine')[:-4] + '_labelIds.png'
        in_name = self.files[idx].replace('leftImg8bit', 'gtFine')[:-4] + '_instanceIds.png'
        dp_name = self.files[idx].replace('leftImg8bit', 'disparity')[:-4] + '.png'


        I = PIL.Image.open(im_name)
        sI = I.resize((self.inst_width, self.inst_height), PIL.Image.ANTIALIAS)
        S = PIL.Image.open(sg_name)
        sS = S.resize((self.inst_width, self.inst_height), PIL.Image.NEAREST)
        N = PIL.Image.open(in_name)
        sN = N.resize((self.inst_width, self.inst_height), PIL.Image.NEAREST)

        D = PIL.Image.open(dp_name)
        dI = np.array(D.resize((self.inst_width, self.inst_height), PIL.Image.BICUBIC))
        dI = dI.astype('float') * self.max_out / self.max_depth
        dI = np.clip(dI, 0, self.max_out)
        depth_mask = dI > self.depth_threshold

        aI = np.array(sI, dtype=np.uint8)
        if self.augment:
            augmented = self.transform(image=aI)
            aI = augmented['image']

        aS = np.array(sS)
        aN = np.array(sN)
        aN[aN < 1000] = 0

        aS_replaced = self.replace_ids(aS)
        aN_c, aN_o = self.get_instances(aN)

        sample = {'image': np.moveaxis(aI, 2, 0).astype('float') / 255.0, 'semseg': aS_replaced, 
                'depth': dI, 'depth_mask': depth_mask, 'inst-center': aN_c, 'inst-offsets': aN_o}

        return sample

    def get_instances(self, inst_seg):
        center = np.zeros((1, self.inst_height, self.inst_width), dtype=np.float32)
        offset = np.zeros((2, self.inst_height, self.inst_width), dtype=np.float32)
        y_coord = np.ones_like(inst_seg, dtype=np.float32)
        x_coord = np.ones_like(inst_seg, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        for seg_id in np.unique(inst_seg)[1:]:
            mask_index = np.where(inst_seg == seg_id)
            center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
            y, x = int(center_y), int(center_x)
            ul = int(np.round(x - 3 * self.sigma - 1)), int(np.round(y - 3 * self.sigma - 1))
            br = int(np.round(x + 3 * self.sigma + 2)), int(np.round(y + 3 * self.sigma + 2))
            c, d = max(0, -ul[0]), min(br[0], self.inst_width) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.inst_height) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.inst_width)
            aa, bb = max(0, ul[1]), min(br[1], self.inst_height)
            center[0, aa:bb, cc:dd] = np.maximum(center[0, aa:bb, cc:dd], self.g[a:b, c:d])

            offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
            offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
            offset[offset_y_index] = center_y - y_coord[mask_index]
            offset[offset_x_index] = center_x - x_coord[mask_index]
        
        return center, offset


class TrainFullAugmentData(torch.utils.data.Dataset):
    def __init__(self, opts, sigma=8):
        super(TrainFullAugmentData, self).__init__()
        file_path = os.path.join(opts.cityscapes_dir, 'leftImg8bit', 'train')
        self.files = self.get_city_files(file_path)
        print('No. of Training Images : ', len(self.files))
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.inst_height = opts.image_height
        self.inst_width = opts.image_width
        self.max_depth = opts.max_depth_dataset
        self.max_out = opts.max_depth
        self.depth_threshold = opts.depth_threshold
        self.augment = opts.augmentations
        if self.augment:
            import albumentations as A
            self.transform = A.Compose([
                                A.RandomCrop(height=opts.image_height, width=opts.image_width, always_apply=True, p=1),
                                A.HorizontalFlip(p=0.25),
                                A.Rotate(limit=10, interpolation=cv2.INTER_CUBIC, p=0.25),
                                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),                                     
                                ], p=1)
    
    @staticmethod
    def get_city_files(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    file_name = os.path.join(r, file)
                    files.append(file_name)
        
        return files
    
    @staticmethod
    def replace_ids(n_arr):
        arr = 19 * np.ones(n_arr.shape, dtype=n_arr.dtype)

        ids = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
            25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for i in ids.keys():
            if i in ids:
                arr[n_arr == i] = ids[i]

        return arr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_name = self.files[idx]
        sg_name = self.files[idx].replace('leftImg8bit', 'gtFine')[:-4] + '_labelIds.png'
        in_name = self.files[idx].replace('leftImg8bit', 'gtFine')[:-4] + '_instanceIds.png'
        dp_name = self.files[idx].replace('leftImg8bit', 'disparity')[:-4] + '.png'


        I = PIL.Image.open(im_name)
        S = PIL.Image.open(sg_name)
        N = PIL.Image.open(in_name)
        D = PIL.Image.open(dp_name)
        aI = np.array(I, dtype=np.uint8)
        aS = np.array(S)
        aN = np.array(N)
        aD = np.array(D)

        if self.augment:
            augmented = self.transform(image=aI, masks=[aS, aN, aD])
            aI = augmented['image']
            aS = augmented['masks'][0]
            aN = augmented['masks'][1]
            aD = augmented['masks'][2]

        dI = aD.astype('float') * self.max_out / self.max_depth
        dI = np.clip(dI, 0, self.max_out)
        depth_mask = dI > self.depth_threshold

        aN[aN < 1000] = 0        
        aS_replaced = self.replace_ids(aS)
        aN_c, aN_o = self.get_instances(aN)

        sample = {'image': np.moveaxis(aI, 2, 0).astype('float') / 255.0, 'semseg': aS_replaced, 
                'depth': dI, 'depth_mask': depth_mask, 'inst-center': aN_c, 'inst-offsets': aN_o}

        return sample

    def get_instances(self, inst_seg):
        center = np.zeros((1, self.inst_height, self.inst_width), dtype=np.float32)
        offset = np.zeros((2, self.inst_height, self.inst_width), dtype=np.float32)
        y_coord = np.ones_like(inst_seg, dtype=np.float32)
        x_coord = np.ones_like(inst_seg, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        for seg_id in np.unique(inst_seg)[1:]:
            mask_index = np.where(inst_seg == seg_id)
            center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
            y, x = int(center_y), int(center_x)
            ul = int(np.round(x - 3 * self.sigma - 1)), int(np.round(y - 3 * self.sigma - 1))
            br = int(np.round(x + 3 * self.sigma + 2)), int(np.round(y + 3 * self.sigma + 2))
            c, d = max(0, -ul[0]), min(br[0], self.inst_width) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.inst_height) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.inst_width)
            aa, bb = max(0, ul[1]), min(br[1], self.inst_height)
            center[0, aa:bb, cc:dd] = np.maximum(center[0, aa:bb, cc:dd], self.g[a:b, c:d])

            offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
            offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
            offset[offset_y_index] = center_y - y_coord[mask_index]
            offset[offset_x_index] = center_x - x_coord[mask_index]
        
        return center, offset


class ValData(torch.utils.data.Dataset):
    def __init__(self, opts, sigma=8):
        super(ValData, self).__init__()
        file_path = os.path.join(opts.cityscapes_dir, 'leftImg8bit', 'val')
        self.files = self.get_city_files(file_path)
        print('No. of Validation Images : ', len(self.files))
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.inst_height = opts.val_image_height
        self.inst_width = opts.val_image_width
        self.max_depth = opts.max_depth_dataset
        self.max_out = opts.max_depth
        self.depth_threshold = opts.depth_threshold
       
    @staticmethod
    def get_city_files(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    file_name = os.path.join(r, file)
                    files.append(file_name)
        
        return files
    
    @staticmethod
    def replace_ids(n_arr):
        arr = 19 * np.ones(n_arr.shape, dtype=n_arr.dtype)

        ids = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
            25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for i in ids.keys():
            if i in ids:
                arr[n_arr == i] = ids[i]

        return arr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_name = self.files[idx]
        sg_name = self.files[idx].replace('leftImg8bit', 'gtFine')[:-4] + '_labelIds.png'
        in_name = self.files[idx].replace('leftImg8bit', 'gtFine')[:-4] + '_instanceIds.png'
        dp_name = self.files[idx].replace('leftImg8bit', 'disparity')[:-4] + '.png'


        I = PIL.Image.open(im_name)
        sI = I.resize((self.inst_width, self.inst_height), PIL.Image.ANTIALIAS)
        S = PIL.Image.open(sg_name)
        sS = S.resize((self.inst_width, self.inst_height), PIL.Image.NEAREST)
        N = PIL.Image.open(in_name)
        sN = N.resize((self.inst_width, self.inst_height), PIL.Image.NEAREST)

        D = PIL.Image.open(dp_name)
        dI = np.array(D.resize((self.inst_width, self.inst_height), PIL.Image.BICUBIC))
        dI = dI.astype('float') * self.max_out / self.max_depth
        dI = np.clip(dI, 0, self.max_out)
        depth_mask = dI > self.depth_threshold

        aI = np.array(sI)
        aS = np.array(sS)
        aN = np.array(sN)
        aN[aN < 1000] = 0

        aS_replaced = self.replace_ids(aS)
        aN_c, aN_o = self.get_instances(aN)

        sample = {'image': np.moveaxis(aI, 2, 0).astype('float') / 255.0, 'semseg': aS_replaced, 
                'depth': dI, 'depth_mask': depth_mask, 'inst-center': aN_c, 'inst-offsets': aN_o}

        return sample

    def get_instances(self, inst_seg):
        center = np.zeros((1, self.inst_height, self.inst_width), dtype=np.float32)
        offset = np.zeros((2, self.inst_height, self.inst_width), dtype=np.float32)
        y_coord = np.ones_like(inst_seg, dtype=np.float32)
        x_coord = np.ones_like(inst_seg, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        for seg_id in np.unique(inst_seg)[1:]:
            mask_index = np.where(inst_seg == seg_id)
            center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
            y, x = int(center_y), int(center_x)
            ul = int(np.round(x - 3 * self.sigma - 1)), int(np.round(y - 3 * self.sigma - 1))
            br = int(np.round(x + 3 * self.sigma + 2)), int(np.round(y + 3 * self.sigma + 2))
            c, d = max(0, -ul[0]), min(br[0], self.inst_width) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.inst_height) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.inst_width)
            aa, bb = max(0, ul[1]), min(br[1], self.inst_height)
            center[0, aa:bb, cc:dd] = np.maximum(center[0, aa:bb, cc:dd], self.g[a:b, c:d])

            offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
            offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
            offset[offset_y_index] = center_y - y_coord[mask_index]
            offset[offset_x_index] = center_x - x_coord[mask_index]
        
        return center, offset


