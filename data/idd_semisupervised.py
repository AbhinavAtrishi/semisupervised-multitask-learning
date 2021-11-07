import os
import torch
import PIL
import numpy as np
from absl import flags

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

flags.DEFINE_string('idd_dir', None, 'IDD Data Directory')

flags.DEFINE_integer('max_depth_dataset', 32000, 'Maximum Depth in the dataset, values above will be clipped')
flags.DEFINE_float('max_depth', 32.0, 'Maximum Depth of the output, values above will be clipped')
flags.DEFINE_float('depth_threshold', 1.0, 'Threshold of Depth in the dataset, values below wont count in the loss')

flags.mark_flag_as_required('idd_dir')
opts = flags.FLAGS

class TrainData(torch.utils.data.Dataset):
    def __init__(self, opts, sigma=8):
        super(TrainData, self).__init__()
        file_path = os.path.join(opts.idd_dir, 'leftImg8bit', 'train')
        self.files = self.get_idd_files(file_path)
        print('Training Images : ', len(self.files))
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.inst_height = opts.val_image_height
        self.inst_width = opts.val_image_width

    @staticmethod
    def get_idd_files(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    file_name = os.path.join(r, file)
                    files.append(file_name)
        
        return files
    
    @staticmethod
    def replace_ids_seg(n_arr):
        arr = 19 * np.ones(n_arr.shape, dtype=n_arr.dtype)
        
        ids = {0:0, 1:0, 2:1, 3:9, 4:11, 5:12, 6:17, 7:18, 8:13, 9:13, 10:14, 11:15, 13:1,
            14:3, 15:4, 18:7, 19:6, 20:5, 22:2, 24:8, 25:10}

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
        sg_name = self.files[idx].replace('leftImg8bit', 'gtFine')[:-4] + '_labellevel3Ids.png'
        in_name = self.files[idx].replace('leftImg8bit', 'gtFine')[:-4] + '_instancelevel3Ids.png'
        
        I = PIL.Image.open(im_name)
        sI = I.resize((self.inst_width, self.inst_height), PIL.Image.ANTIALIAS)
        S = PIL.Image.open(sg_name)
        sS = S.resize((self.inst_width, self.inst_height), PIL.Image.NEAREST)
        N = PIL.Image.open(in_name)
        sN = N.resize((self.inst_width, self.inst_height), PIL.Image.NEAREST)
        
        aI = np.array(sI, dtype=np.uint8)
        aS = np.array(sS)
        aN = np.array(sN)
        aN[aN < 1000] = 0

        hedI = cv2.cvtColor(aI.copy(), cv2.COLOR_RGB2BGR)

        aS_replaced = self.replace_ids_seg(aS)
        aN_c, aN_o = self.get_instances(aN)

        sample = {'image': np.moveaxis(aI, 2, 0).astype('float') / 255.0, 'semseg': aS_replaced, 
                 'inst-centers': aN_c, 'inst-offsets': aN_o, 'hed_image': np.moveaxis(hedI, 2, 0)}

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
        file_path = os.path.join(opts.idd_dir, 'leftImg8bit', 'val')
        self.files = self.get_idd_files(file_path)
        print('Validation Images : ', len(self.files))
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.inst_height = opts.val_image_height
        self.inst_width = opts.val_image_width

    @staticmethod
    def get_idd_files(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    file_name = os.path.join(r, file)
                    files.append(file_name)
        
        return files
    
    @staticmethod
    def replace_ids_seg(n_arr):
        arr = 19 * np.ones(n_arr.shape, dtype=n_arr.dtype)
        
        ids = {0:0, 1:0, 2:1, 3:9, 4:11, 5:12, 6:17, 7:18, 8:13, 9:13, 10:14, 11:15, 13:1,
            14:3, 15:4, 18:7, 19:6, 20:5, 22:2, 24:8, 25:10}

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
        sg_name = self.files[idx].replace('leftImg8bit', 'gtFine')[:-4] + '_labellevel3Ids.png'
        in_name = self.files[idx].replace('leftImg8bit', 'gtFine')[:-4] + '_instancelevel3Ids.png'
        
        I = PIL.Image.open(im_name)
        sI = I.resize((self.inst_width, self.inst_height), PIL.Image.ANTIALIAS)
        S = PIL.Image.open(sg_name)
        sS = S.resize((self.inst_width, self.inst_height), PIL.Image.NEAREST)
        N = PIL.Image.open(in_name)
        sN = N.resize((self.inst_width, self.inst_height), PIL.Image.NEAREST)
        
        aI = np.array(sI)
        aS = np.array(sS)
        aN = np.array(sN)
        aN[aN < 1000] = 0

        aS_replaced = self.replace_ids_seg(aS)
        aN_c, aN_o = self.get_instances(aN)

        sample = {'image': np.moveaxis(aI, 2, 0).astype('float') / 255.0, 'semseg': aS_replaced, 
                 'inst-centers': aN_c, 'inst-offsets': aN_o}

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
