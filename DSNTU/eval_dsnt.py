import torch
from tqdm import tqdm
import dsntnn
from INFO import sigma_t


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 #if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = torch.nn.L1Loss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)

            true_masks = true_masks.to(device=device, dtype=mask_type)
            with torch.no_grad():
                coords,heatmap = net(imgs)

            if net.n_classes > 1:
                loss1 = dsntnn.euclidean_losses(coords, true_masks)
                loss2 = dsntnn.js_reg_losses(heatmap, true_masks, sigma_t=sigma_t)
                loss = dsntnn.average_loss(loss1 + loss2)
                tot += loss.item()
            else:
                tot += criterion(coords, true_masks).item()/imgs.shape[0]


            pbar.update()

    net.train()
    if n_val!=0:
        return tot/n_val
    else:
        return tot


