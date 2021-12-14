import torch
import numpy as np
import torch.nn as nn
from PIL import Image, ImageDraw
# from sklearn.cluster import DBSCAN

def target2line(target, img_size, k, eval=False):
    """
    target: line representetitve in grid [L, grid_h, grid_w]
    img_size: (width, height): Input image size, PIL Image size

    eval=False : Default. For inference. Line width not big.
    eval=True : For iou. Line width is bigger.

    return line_img
    """
    line_img = Image.new("L", img_size)
    draw = ImageDraw.Draw(line_img)
    resolution = 32 / pow(2, k)
    grid_h = int(img_size[1] // resolution)
    grid_w = int(img_size[0] // resolution)
    line_width = 4

    if not eval:
        line_width = 2
        for i in range(1, grid_h):
            grid = []
            grid.append(((0, i * img_size[1]/grid_h)) )
            grid.append(((img_size[0], i * img_size[1]/grid_h)) )
            draw.line(grid, fill='blue', width=0)
        for i in range(grid_w):
            grid = []
            grid.append(((i * img_size[0]/grid_w, 0)) )
            grid.append(((i * img_size[0]/grid_w, img_size[1])) )
            draw.line(grid, fill='blue', width=0)
    
    targets = np.transpose(target, [1, 2, 0])
    targets = targets.reshape(grid_h, grid_w, -1, 4)

    offset_x = np.linspace(0, grid_w - 1, grid_w)
    offset_y = np.linspace(0, grid_h - 1, grid_h)
    off_w, off_h = np.meshgrid(offset_x, offset_y)

    indexes = np.argwhere( np.sum( targets.reshape(-1, 4) , axis=1, keepdims=False ) > 0 )
    targets = np.transpose(targets, (3, 2, 0, 1))

    targets[0,:] += off_w
    targets[1,:] += off_h
    targets[2,:] += off_w
    targets[3,:] += off_h

    targets = (targets * resolution) 
    targets = np.transpose(targets, ( 2, 3, 1, 0)).reshape(-1, 4)
   
    detected = targets[indexes[..., 0]]
    # print( 'detected lines shape: ', detected.shape)
    [draw.line([(x1, y1), (x2,y2)], fill='white', width=line_width) for (x1, y1, x2, y2) in detected ]
    return line_img

def iou_v2(target, pred, img_size, k, conf_thresh):
    """
    target.shape = [bs, L * 4, grid_h, grid_w]
    pred.shape = [bs, P * 6, grid_h, grid_w]

    IoU will be calculated from target img and prediction img.

    return IoU
    """

    bs, P, grid_h, grid_w = pred.shape
    P = P // 6
    resolution = 32 // pow(2, k)
    sigmoid = nn.Sigmoid()
    pred = torch.reshape(pred, (bs, P, 6, grid_h, grid_w))
    pred = pred * (pred[:, :, 4:5, :, :] > conf_thresh ) * (sigmoid(pred[:, :, 5:6, :, :]) > 0.5 )
    pred = torch.reshape( pred[:, :, 0:4, :, :], (bs, P * 4, grid_h, grid_w))

    iou = 0
    for i in range(bs):
        target_img = target2line(target[i, ...].detach().cpu().squeeze().numpy(), img_size=img_size, k=k, eval=True)
        pred_img = target2line(pred[i, ...].detach().cpu().squeeze().numpy(), img_size=img_size, k=k, eval=True)
        
        target_arr = np.array(target_img, dtype=np.uint8)
        pred_arr = np.array(pred_img, dtype=np.uint8)

        inter = pred_arr[target_arr > 0] > 0
        union = (target_arr > 0) + (pred_arr > 0)
        iou += np.sum(inter) / (np.sum(union)+ 1e-6)
    return iou / bs

def inference(pred, img_size, k, conf_thresh):
    """
    target.shape = [bs, L * 4, grid_h, grid_w]
    pred.shape = [bs, P * 6, grid_h, grid_w]

    return inference image
    """
    bs, P, grid_h, grid_w = pred.shape
    P = P // 6
    # resolution = 32 // pow(2, k)
    sigmoid = nn.Sigmoid()
    pred = torch.reshape(pred, (bs, P, 6, grid_h, grid_w))
    pred = pred * (pred[:, :, 4:5, :, :] > conf_thresh ) * (sigmoid(pred[:, :, 5:6, :, :]) > 0.5 )
    pred = torch.reshape( pred[:, :, 0:4, :, :], (bs, P * 4, grid_h, grid_w))
    return target2line(pred.cpu().detach().squeeze().numpy(), img_size=img_size, k=k, eval=True)

class YolinoLoss(nn.Module):
    def __init__(self, weighted=False):
        super(YolinoLoss, self).__init__()

    def forward(self, target, pred, eval=False):
        """
        loss function from "YOLinO: arXiv:2103.14420v1 [cs.CV] 26 Mar 2021 "

        target.shape = [batch_size, line_info(L*4), grid_h, grid_w]
        pred: [batch_size, line_info(P*6), grid_size, grid_size]

        return loss
        """
        loss = 0
        bs = target.shape[0]
        L = target.shape[1] // 4
        P = pred.shape[1] // 6
        grid_h = target.shape[2]
        grid_w = target.shape[3]

        target = torch.reshape(target, (bs, L, 4, grid_h, grid_w))
        pred = torch.reshape(pred, (bs, P, 6, grid_h, grid_w))

        target = torch.reshape( torch.permute(target, (0, 3, 4, 1, 2)), (-1, L, 4)).unsqueeze(2)
        pred = torch.reshape( torch.permute(pred, (0, 3, 4, 1, 2)), (-1, P, 6)).unsqueeze(1)

        loc_dist_matrix = torch.sum( torch.sqrt( torch.sum( torch.square( torch.reshape(target, (-1,L,1,2,2)) - torch.reshape(pred[..., 0:4], (-1,1,P,2,2))), dim=-1)), dim=-1)    
        loc_loss_matrix, _ = torch.min(loc_dist_matrix, dim=1, keepdim=True)
        thresh = torch.ones_like(loc_loss_matrix) * -1
        loc_loss_matrix_thresh = torch.where(loc_loss_matrix < 2, loc_loss_matrix, thresh)
        target_line_matrix = torch.sum(target , dim=-1) > 0
        
        ijk_mask = (loc_dist_matrix == loc_loss_matrix_thresh) * target_line_matrix
        ik_mask, _ = torch.max(ijk_mask, dim=1, keepdim=True)
        sigmoid = nn.Sigmoid()
        
        # location loss
        loss += torch.sum( loc_dist_matrix[ijk_mask])   
        # resp loss  
        loss += torch.sum( torch.square( sigmoid(pred[..., 4:5]) - 1)[ik_mask] )
        # no_resp loss
        loss += torch.sum( torch.square( sigmoid(pred[..., 4:5]) )[~ik_mask] )
        # class loss
        loss += torch.sum( torch.square( torch.ones_like(target[..., :1]) - (sigmoid(pred[..., 5:6]) > 0.5) * 1 )[ijk_mask] )

        return loss / bs






# def pred_img(predd, img_size, k):
#     """
#     receive prediction from NMSwithDBSCAN
#     predd shape = [:, (mx, my, l, dx, dy)]

#     return img
#     """
#     resolution = 32 // pow(2, k)
#     pred_img = Image.new('L', img_size )
#     draw = ImageDraw.Draw(pred_img)
#     xy1 = predd[..., 0:2] - predd[..., 2:3] * predd[..., 3:4] / 2
#     xy2 = predd[..., 0:2] + predd[..., 2:3] * predd[..., 3:4] / 2

#     _predd = np.concatenate( (xy1, xy2), axis=1 ) * resolution
   
#     [ draw.line([(x1, y2), (x2,y1)], fill='white', width=2) for (x1,y1, x2, y2) in  _predd  ]
#     return pred_img

# def NMSwithDBSCAN(prediction):
#     """
#     P(g, l, c): 
#     P: Predictor 
#     g: geometry representation of line segment
#     l: class confidence
#     c: confidence of predictor


#     discard all predictors with c <= t_c

#     g_dash = (mx, my, l, dx, dy) mid point coord = mx, my
#     normalized directions=dx, dy

#     return pred_result : [-1, (mx, my, l, dx, dy)] 
#     """
#     # temporary input   
#     # prediction = torch.clamp(torch.randn(2, 48, 28, 60), min=0, max=1 )  # between 0 ~ 1
#     # prediction = torch.clamp(torch.randn(2, 6, 2, 2), min=0, max=1 )  # between 0 ~ 1


#     k = 1
#     lambda_m = 2
#     lambda_l = 0.013
#     lambda_d = 0.05
#     epss = 0.02
#     thresh_conf = 0.9
#     # resolution = 32 / pow(2, k)
#     bs, P, grid_h, grid_w = prediction.shape
#     # P = P // 6
#     P = P // 5 # jikken you 

#     # ppred = torch.reshape( prediction, (bs, P, 6, grid_h, grid_w) )
#     ppred = torch.reshape( prediction, (bs, P, 5, grid_h, grid_w) )    #jikken you
    
#     # prediction confidence higher than thresh_conf
#     high_conf_mask = (ppred[:, :, 4:5, :, :] >= thresh_conf) 

#     # conversion
#     h, w = torch.arange(grid_h), torch.arange(grid_w)
#     offset_h, offset_w = torch.meshgrid(h, w, indexing='ij')

#     # g(x, y) -> g_hat(x_hat, y_hat)
#     ppred[:, :, 0, :, :] += offset_w # x1
#     ppred[:, :, 1, :, :] += offset_h # y1
#     ppred[:, :, 2, :, :] += offset_w # x2
#     ppred[:, :, 3, :, :] += offset_h # y2

#     # mx, my 
#     m_xy = (ppred[:, :, 2:4, :, :] + ppred[:, :, 0:2, :, :])  * k * 0.5 #* lambda_m
#     diff = ppred[:, :, 2:4, :, :] - ppred[:, :, 0:2, :, :]
#     l = torch.sqrt( torch.square(diff[:, :, 0, :, :] ) + torch.square(diff[:, :, 1, :, :] ) )# * lambda_l
#     dx = torch.square(diff[:, :, 0, :, :]) / (l + 1e-6) #* lambda_d 
#     dy = torch.square(diff[:, :, 1, :, :])  / (l + 1e-6) #* lambda_d
    
#     db_pred = torch.cat((m_xy, l.unsqueeze(2), dx.unsqueeze(2), dy.unsqueeze(2) ), dim=2)
#     db_pred = torch.reshape(torch.permute(db_pred, (0,1,3,4,2)), (-1, 5))

#     # DBSCAN 
#     db_pred = db_pred.cpu().numpy()
#     conffs = high_conf_mask * (ppred[:, :, 4:5, :, :] > 0)
#     conffs = torch.reshape( torch.permute(conffs, (0,1,3,4,2)),(-1, 1)).cpu().numpy().squeeze(1)
#     """
#     weights = torch.pow(torch.reshape( torch.permute(ppred[:, :, 4:5, :, :], (0,1,3,4,2)),(-1, 1)), 10 ).numpy().squeeze()
#     db_result = DBSCAN(eps=epss, min_samples=2).fit(db_pred, sample_weight=weights)
#     labels = db_result.labels_

#     print( len(np.unique(labels)) - 1 )

#     # Select Max score 
#     # labels[3] = 0
#     # labels[5] = 1
#     # labels[0] = 1
#     # db_pred = np.clip(np.random.randn(8, 5), a_min=0, a_max=1)

#     nx, ny = (len(np.unique(labels)) - 1, len(labels))
#     x = np.linspace(0, nx - 1, nx)
#     y = np.linspace(0, ny - 1, ny)

#     xv, yv =np.meshgrid(x, y)
#     labels_matrix = np.zeros_like(xv) + np.expand_dims(labels, 1)

#     averaging_mask = labels_matrix == xv
#     big_mask_matrix = np.zeros( (db_pred.shape[1], averaging_mask.shape[0], averaging_mask.shape[1] ) )
#     big_mask_matrix += averaging_mask
#     big_mask_matrix = (big_mask_matrix > 0) * conffs

#     max_conf_matrix = np.max(big_mask_matrix, axis=1, keepdims=True)
#     big_max_mask = (max_conf_matrix == big_mask_matrix) * (big_mask_matrix > 0)
#     big_max_mask = np.transpose(big_max_mask, (2, 1, 0))
    
#     pred = big_max_mask * db_pred
#     non_zero = np.sum(pred, axis=2, keepdims=False) > 0

#     pred_result = pred[non_zero, :]
#     """
#     print(conffs.shape)
#     print(conffs[50:60])
#     pred_result = db_pred[conffs, :]

#     return pred_result