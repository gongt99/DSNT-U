import argparse
import logging
import os
from os.path import splitext
import numpy as np
import torch
from PIL import Image
from scipy.io import savemat
from unet import UNet_dsnt
from utils.dataset import BasicDataset
import hdf5storage
from scipy.ndimage import zoom
from INFO import dir_test,tasklist,taskname


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

def predict_img(net,
                img,
                device,
                scale_factor=1,):
    net.eval()
    img = zoom(img, [0.5, 0.5, 0.5])
    img_gradient = get_gradient(img)
    img = (img - np.min(img)) / (np.max(img)-np.min(img))
    img_gradient = (img_gradient - np.min(img_gradient)) / (np.max(img_gradient) - np.min(img_gradient))
    img = BasicDataset.preprocess(img, scale_factor)
    img_gradient = BasicDataset.preprocess(img_gradient, scale_factor)
    img_concat = np.concatenate([img, img_gradient], axis=0)
    img = torch.from_numpy(img_concat)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        coords,heatmap = net(img)
        coords = coords
        heatmap = heatmap

    return coords,heatmap



def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files.item()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    in_files = dir_test
    for j in range(len(tasklist)):
        task = tasklist[j]
        model = f'./checkpoints' + '/' + taskname + '/' + task + '/best_model.pth'
        pred_path = in_files
        #out_files = get_output_filenames(args)
        pred = os.listdir(pred_path)
        net = UNet_dsnt(n_channels=2, n_classes=3)
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Loading model {}".format(model))
        device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        net.to(device=device)
        net.load_state_dict(torch.load(model, map_location=device))
        logging.info("Model loaded !")
        dice_mean = 0
        mask_save = np.zeros(1)
        true_mask_save = np.zeros(1)

        for i, fn in enumerate(pred):
            logging.info("\nPredicting image {} ...".format(fn))
            subjectname = splitext(fn)[0]
            data_h5py = hdf5storage.loadmat(os.path.join(pred_path,fn))
            img = data_h5py['output']['data'][0][0]

            papx = (data_h5py['output']['papx'][0][0]-1)*[0.5,0.5,0.5]
            pend = (data_h5py['output']['pend'][0][0]-1)*[0.5,0.5,0.5]
            pright = (data_h5py['output']['pright'][0][0]-1)*[0.5,0.5,0.5]

            true_mask = np.hstack((papx, pend, pright))

            coords,heatmap = predict_img(net=net,
                               img=img,
                               scale_factor=1,
                               device=device)
            coords = coords.cpu().numpy().squeeze()
            coords = coords + 1
            mask = np.array([coords[0, 0] * 81-0.5, coords[0, 1] * 81-0.5, coords[0, 2] * 29.25
                             -0.5, coords[1, 0] * 81-0.5, coords[1, 1] * 81-0.5,
                             coords[1, 2] * 29.25-0.5, coords[2, 0] * 81-0.5, coords[2, 1] * 81-0.5, coords[2, 2] * 29.25-0.5])

            logging.info("pred result: {},\ntrue mask:{}".format(mask,true_mask))
            outpath = '/home/bit717-3/Pycharmproject/Caridac_dsnt_20230302/predict/'+ taskname + '/'  + task
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            savemat(outpath + '/'+subjectname+'.mat',{'pred': mask, 'true_mask': true_mask, 'heatmap': heatmap})
