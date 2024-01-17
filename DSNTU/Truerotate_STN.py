import numpy as np
import math
import torch
import torch.nn.functional as F

def getpatch(data,papx,pend,pright,patchsize):

    patch = np.zeros(data.shape)
    patch[int(round(papx[1]) - patchsize[0, 0] / 2):int(round(papx[1]) + patchsize[0, 0] / 2),
                 int(round(papx[0]) - patchsize[0, 1] / 2):int(round(papx[0]) + patchsize[0, 1] / 2),
                 int(round(papx[2]) - patchsize[0, 2] / 2):int(round(papx[2]) + patchsize[0, 2] / 2)] = 1
    patch[int(round(pend[1]) - patchsize[1, 0] / 2):int(round(pend[1]) + patchsize[1, 0] / 2),
                 int(round(pend[0]) - patchsize[1, 1] / 2):int(round(pend[0]) + patchsize[1, 1] / 2),
                 int(round(pend[2]) - patchsize[1, 2] / 2):int(round(pend[2]) + patchsize[1, 2] / 2)] = 1
    patch[int(round(pright[1]) - patchsize[2, 0] / 2):int(round(pright[1]) + patchsize[2, 0] / 2),
                   int(round(pright[0]) - patchsize[2, 1] / 2):int(round(pright[0]) + patchsize[2, 1] / 2),
                   int(round(pright[2]) - patchsize[2, 2] / 2):int(round(pright[2]) + patchsize[2, 2] / 2)]=1

    return patch
def transform3D(image, affine_matrix, papx, pend, pright):

    B, H, W, D, C = image.shape
    M = affine_matrix

    # mesh grid generation
    x = np.linspace(0, 1, W) * (W - 1)
    y = np.linspace(0, 1, H) * (H - 1)
    z = np.linspace(0, 1, D) * (D - 1)
    footpoint = (pend+papx)/2
    x = x - footpoint[0]
    y = y - footpoint[1]
    z = z - footpoint[2]
    x_t, y_t, z_t = np.meshgrid(x, y, z)



    ones = np.ones(np.prod(x_t.shape))
    sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), z_t.flatten(), ones])

    sampling_grid = np.resize(sampling_grid, (B, 4, H * W * D))

    batch_grids = np.matmul(M, sampling_grid)  # the batch grid has the shape (B, 3, H*W*D)

    batch_grids = batch_grids.reshape(B, 3, H, W, D)
    batch_grids = np.moveaxis(batch_grids, 1, -1)

    # bilinear resampler
    x_s = batch_grids[:, :, :, :, 0:1].squeeze()
    y_s = batch_grids[:, :, :, :, 1:2].squeeze()
    z_s = batch_grids[:, :, :, :, 2:3].squeeze()

    targetpoint = [162, 162, 58]
    bias1 = getbias1(footpoint, M, targetpoint)
    newpapx = getnewpoint(papx, M, bias1, footpoint)
    newpend = getnewpoint(pend, M, bias1, footpoint)
    newpright = getnewpoint(pright, M, bias1, footpoint)
    x = (x_s) + bias1[0]
    y = (y_s) + bias1[1]
    z = (z_s) + bias1[2]

    # for each coordinate we need to grab the corner coordinates
    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64)
    y1 = y0 + 1
    z0 = np.floor(z).astype(np.int64)
    z1 = z0 + 1

    # clip to fit actual image size
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)
    z0 = np.clip(z0, 0, D - 1)
    z1 = np.clip(z1, 0, D - 1)

    # grab the pixel value for each corner coordinate
    Ia = image[np.arange(B)[:, None, None, None], y0, x0, z0]
    Ib = image[np.arange(B)[:, None, None, None], y1, x0, z0]
    Ic = image[np.arange(B)[:, None, None, None], y0, x1, z0]
    Id = image[np.arange(B)[:, None, None, None], y1, x1, z0]
    Ie = image[np.arange(B)[:, None, None, None], y0, x0, z1]
    If = image[np.arange(B)[:, None, None, None], y1, x0, z1]
    Ig = image[np.arange(B)[:, None, None, None], y0, x1, z1]
    Ih = image[np.arange(B)[:, None, None, None], y1, x1, z1]

    # calculated the weighted coefficients and actual pixel value
    wa = (x1 - x) * (y1 - y) * (z1 - z)
    wb = (x1 - x) * (y - y0) * (z1 - z)
    wc = (x - x0) * (y1 - y) * (z1 - z)
    wd = (x - x0) * (y - y0) * (z1 - z)
    we = (x1 - x) * (y1 - y) * (z - z0)
    wf = (x1 - x) * (y - y0) * (z - z0)
    wg = (x - x0) * (y1 - y) * (z - z0)
    wh = (x - x0) * (y - y0) * (z - z0)

    # add dimension for addition
    wa = np.expand_dims(wa, axis=0)
    wb = np.expand_dims(wb, axis=0)
    wc = np.expand_dims(wc, axis=0)
    wd = np.expand_dims(wd, axis=0)
    we = np.expand_dims(we, axis=0)
    wf = np.expand_dims(wf, axis=0)
    wg = np.expand_dims(wg, axis=0)
    wh = np.expand_dims(wh, axis=0)
    wa = np.expand_dims(wa, axis=4)
    wb = np.expand_dims(wb, axis=4)
    wc = np.expand_dims(wc, axis=4)
    wd = np.expand_dims(wd, axis=4)
    we = np.expand_dims(we, axis=4)
    wf = np.expand_dims(wf, axis=4)
    wg = np.expand_dims(wg, axis=4)
    wh = np.expand_dims(wh, axis=4)

    # compute output
    image_out = wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih
    #image_out = image_out.astype(np.int64)

    return image_out, newpapx, newpend, newpright

def affine_transform(data,papx,pend,pright):
    # 3D
    layer_num = 8
    #input_img = load3D(layer_num)
    input_img = data
    footpoint = pend
    m = pend[0] - papx[0]
    n = pend[1] - papx[1]
    s = pend[2] - papx[2]
    a = papx[0]
    b = papx[1]
    c = papx[2]
    t = -(m * (a - pright[0]) + n * (b - pright[1]) + s * (c - pright[2])) / (m * m + n * n + s * s)
    pr0 = [(m * t + a), (n * t + b), (s * t + c)]
    #pr1 = [0., 0., 0.]
    zold = [m, n, s] / np.sqrt(m * m + n * n + s * s)
    xold = [-pright[0] + pr0[0], -pright[1] + pr0[1], -pright[2] + pr0[2]] / np.sqrt(
        (-pright[0] + pr0[0]) ** 2 + (-pright[1] + pr0[1]) ** 2 + (-pright[2] + pr0[2]) ** 2)
    yold = np.cross(zold, xold)
    T = np.array([xold, yold, zold])
    T = T.transpose()
    t = np.zeros((4, 4))
    t[:3, :3] = T
    pr0.append(1.)
    eularx = math.atan2(t[1][2], t[2][2])
    eularz = math.atan2(-t[0][2], math.sqrt(T[0][0] ** 2 + T[0][1] ** 2))
    eulary = math.atan2(t[0][1], t[0][0])
    t = t[0:3, 0:4];
    # define the affine matrix
    # initialize M to identity transform
    M = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.] , [0., 0., 1., 0.]])
    # repeat num_batch times
    M = np.resize(M, (input_img.shape[0], 3, 4))

    # change affine matrix values
    # translation
    M[0,:,:] = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    #img_translate = transform3D(input_img, M)

    # rotation
    alpha =eularx #degree
    beta =eularz
    gamma = eulary
    # Tait-Bryan angles in homogeneous form, reference: https://people.cs.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf
    Rx = [[1,0,0,0],[0,math.cos(alpha),-math.sin(alpha),0],[0,math.sin(alpha),math.cos(alpha),0],[0.,0.,0,1.]]
    Ry = [[math.cos(beta),0,math.sin(beta),0],[0,1,0,0],[-math.sin(beta),0,math.cos(beta),0],[0.,0.,0.,1.]]
    Rz = [[math.cos(gamma),-math.sin(gamma),0,0],[math.sin(gamma),math.cos(gamma),0,0],[0,0,1,0],[0.,0.,0.,1.]]



    M[0,:,:] = np.matmul(Rz,np.matmul(Ry,Rx))[0:3,:]
    M = M[0]

    img_rotate,newpapx,newpend,newpright = transform3D(input_img, t,papx,pend,pright)
    return img_rotate.squeeze(),newpapx,newpend,newpright

def stn(x, papx, pend, pright):
    m = pend[:, 0] - papx[:, 0]
    n = pend[:, 1] - papx[:, 1]
    s = pend[:, 2] - papx[:, 2]
    a = papx[:, 0]
    b = papx[:, 1]
    c = papx[:, 2]
    t = -(m * (a - pright[:, 0]) + n * (b - pright[:, 1]) + s * (c - pright[:, 2])) / (m * m + n * n + s * s)
    pr0 = torch.cat(((m * t + a).unsqueeze(1), (n * t + b).unsqueeze(1), (s * t + c).unsqueeze(1)), 1)
    # pr1 = [0., 0., 0.]
    zold = torch.cat((m.unsqueeze(1), n.unsqueeze(1), s.unsqueeze(1)), 1) / torch.sqrt(m * m + n * n + s * s).unsqueeze(
        1)
    xold = torch.cat(((-pright[:, 0] + pr0[:, 0]).unsqueeze(1), (-pright[:, 1] + pr0[:, 1]).unsqueeze(1),
                      (-pright[:, 2] + pr0[:, 2]).unsqueeze(1)), 1) / torch.sqrt(
        (-pright[:, 0] + pr0[:, 0]) ** 2 + (-pright[:, 1] + pr0[:, 1]) ** 2 + (
                    -pright[:, 2] + pr0[:, 2]) ** 2).unsqueeze(1)
    yold = torch.mul(zold, xold)
    ts = torch.cat(((1 / 2 - (papx[:, 0] + pend[:, 0]) / 2).unsqueeze(1),
                    (1 / 2 - (papx[:, 1] + pend[:, 1])).unsqueeze(1), (1 / 2 - (papx[:, 2] + pend[:, 2])).unsqueeze(1)),
                   1)
    T = torch.cat((xold.unsqueeze(2), yold.unsqueeze(2), zold.unsqueeze(2)), 2)
    T = T.permute(0,2,1)
    T = torch.cat((T,ts.unsqueeze(2)),2)
    grid = F.affine_grid(T, x.shape)
    footpoint = (pend + papx) / 2
    x = F.grid_sample(x, grid,align_corners=True)
    return x,grid


def getnewpoint(point,Matrix,bias1,footpoint):
    new_point = point - bias1
    new_point = np.append(np.array(new_point),1)
    new_point = np.expand_dims(new_point,axis=1)
    M_ni = np.zeros((3,4))
    M_ni[:,0:3] = np.linalg.inv(Matrix[:,0:3])
    new_point = np.matmul(M_ni,new_point)
    new_point = new_point.squeeze() + footpoint
    return new_point

def getbias1(footpoint,Matrix,targetpoint):
    minus = targetpoint - footpoint
    minus = np.append(np.array(minus),1)
    minus = np.expand_dims(minus,axis=1)
    rotminus = np.matmul(Matrix,minus)
    bias = footpoint - rotminus.squeeze()
    return bias

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

def getpatchmask_fromrotimg(T,footpoint,newpapx,targetpoint=[54,54,54]):
    bias = getbias1(footpoint,T,targetpoint)
    papxpatch_mask_x = np.linspace((round(newpapx[0]))-10,(round(newpapx[0]))+10,20)
    papxpatch_mask_y = np.linspace((round(newpapx[1]))-10,(round(newpapx[1]))+10,20)
    papxpatch_mask_z = np.linspace((round(newpapx[2]))-5,(round(newpapx[2]))+5,10)
    papxpatch_mask_x_t, papxpatch_mask_y_t, papxpatch_mask_z_t = np.meshgrid(papxpatch_mask_x, papxpatch_mask_y, papxpatch_mask_z)
    ones = np.ones(np.prod(papxpatch_mask_x_t.shape))
    sampling_grid = np.vstack([papxpatch_mask_x_t.flatten(), papxpatch_mask_y_t.flatten(), papxpatch_mask_z_t.flatten(), ones])
    # repeat to number of batches
    sampling_grid = np.resize(sampling_grid, (4, 20 * 20 * 10))
    M_ni = np.zeros((3, 4))
    M_ni[:, 0:3] = np.linalg.inv(T[:, 0:3])
    batch_grids = np.matmul(M_ni, sampling_grid)
    #3new_point = batch_grids.squeeze()  # the batch grid has the shape (B, 3, H*W*D)
    # new_footpoint = np.matmul(M,footpoint)

    # reshape to (B, H, W, D, 3)
    batch_grids = batch_grids.reshape(3, 20, 20, 10)
    batch_grids = np.moveaxis(batch_grids, 0, -1)

    # bilinear resampler
    x_s = batch_grids[ :, :, :, 0:1].squeeze()
    y_s = batch_grids[:, :, :, 1:2].squeeze()
    z_s = batch_grids[:, :, :, 2:3].squeeze()
    x = y_s + footpoint[0]
    y = x_s + footpoint[1]
    z = z_s + footpoint[2]
    x = np.clip(x,0,107)
    y = np.clip(y,0,107)
    z = np.clip(z, 0, 63)



    return x.astype(np.int), y.astype(np.int), z.astype(np.int)

def getFootPoint(point, line_p1, line_p2):
    """
        @point, line_p1, line_p2 : [x, y, z]
    """
    x0 = point[0]
    y0 = point[1]
    z0 = point[2]

    x1 = line_p1[0]
    y1 = line_p1[1]
    z1 = line_p1[2]

    x2 = line_p2[0]
    y2 = line_p2[1]
    z2 = line_p2[2]

    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / \
         ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)*1.0

    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1
    zn = k * (z2 - z1) + z1

    return (xn, yn, zn)

if __name__=="__main__":
	main()