from os.path import splitext
from os import listdir
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import logging
from scipy.ndimage import zoom
import hdf5storage
import random


def get_gradient(img):
    gradient_x = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
    gradient_y = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    gradient_z = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for i in range(img.shape[0]-1):
        gradient_x[i,:,:] = img[i,:,:]-img[i+1,:,:]
    for j in range(img.shape[1]-1):
        gradient_y[:,j,:] = img[:,j,:]-img[:,j+1,:]
    for k in range(img.shape[2] - 1):
        gradient_z[:, :, k] = img[:, :, k] - img[:, :, k+1]
    gradient = np.sqrt(np.power(gradient_x,2)+np.power(gradient_y,2)+np.power(gradient_z,2))
    return gradient

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.scale = scale
        self.mask_suffix = mask_suffix


    def __len__(self):
        return len(self.ids)
    @classmethod
    def preprocess(cls,pil_img,scale):
        assert len(pil_img.shape) == 3, f'size error'

        img_nd = zoom(pil_img,(scale,scale,1))
        img_nd = np.array(img_nd)
        img_nd = np.expand_dims(img_nd,axis = 3)
          #pil_img = np.expand_dims(pil_img,axis = 4)
        img_trans = img_nd.transpose(3,2,0,1) #CZHW
        return img_trans



    def __getitem__(self, index):
        idx = self.ids[index]
        mask_file = os.path.join(self.masks_dir , idx)
        img_file = os.path.join(self.imgs_dir , idx)
        data_h5py = hdf5storage.loadmat(mask_file+'.mat')
        data_h5py = data_h5py['output']
        petimg = data_h5py['data'][0][0]
        papx = data_h5py['papx'][0][0][0]-1
        pend = data_h5py['pend'][0][0][0]-1
        pright = data_h5py['pright'][0][0][0]-1
        papx_new = np.array(papx)
        pend_new = np.array(pend)
        pright_new = np.array(pright)
        papx_new = (papx_new * 2 + 1) / np.array(petimg.shape) - 1
        pend_new = (pend_new * 2 + 1) / np.array(petimg.shape) - 1
        pright_new = (pright_new * 2 + 1) / np.array(petimg.shape) - 1

        petimg = zoom(petimg, [0.5, 0.5, 0.5])
        petimg_gradient = get_gradient(petimg)
        petimg = (petimg - np.min(petimg)) / (np.max(petimg)-np.min(petimg))
        petimg_gradient = (petimg_gradient - np.min(petimg_gradient)) / (np.max(petimg_gradient) - np.min(petimg_gradient))


        position = np.array([[papx_new],[pend_new],[pright_new]])
        position = position.squeeze()

        petimg_new = self.preprocess(petimg, self.scale)
        petimg_gradient_new = self.preprocess(petimg_gradient, self.scale)
        petimg_concat = np.concatenate([petimg_new, petimg_gradient_new], axis=0)

        return {
            'name': idx,
            'image': torch.from_numpy(petimg_concat).type(torch.FloatTensor),
            'mask': torch.from_numpy(position).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
