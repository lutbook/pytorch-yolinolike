import numpy as np
import os, json, torch, math
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset
# from functions import iou_v2, target2line, YolinoLoss

class YolinoDataset(Dataset):
    def __init__(self, dataset_dir, image_transform, phase, p, k):
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform
        self.phase = phase
        self.p = p
        self.k = k
        labels_json = open(os.path.join(dataset_dir, 'line_label_' + self.phase + '.json'))
        self.labels = json.load(labels_json)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        labels = self.labels[idx]
        img_name = os.path.join(self.dataset_dir, labels['file'] )
        image = Image.open( img_name  ).convert('RGB')
        old_size = image.size
        # target_img = Image.open( img_name.replace('images', 'labels').replace('.png', '_labelLine.png') ).convert('L') # for intuitive check
        if self.image_transform is not None:
            image = self.image_transform(image)
        new_size = (image.shape[2], image.shape[1])
        # target_img = target_img.resize(new_size, Image.NEAREST )    # for intuitive check
        target = torch.from_numpy( poly2cell(labels, old_size, new_size, self.k) )

        if target.shape[0] <= (self.p) * 4:
            target = torch.cat( (target, torch.zeros(self.p * 4 - target.shape[0], target.shape[1], target.shape[2])), dim=0)
        else:
            print('GT_L={} is exceeded predictor head P={}'.format(target.shape[0], self.p))
        return image, target

def poly2cell(label, old_size, new_size, k):
    """
    label: polylines in label
    img_size: original image size, PIL Image size
    k: k=1 at current network archictecture.
        (for further implementation: k=0->cell=32x32, k=1->cell=16x16, k=2->cell=8x8pxls)

    YOLinO line2grid process:
    a) Slice
    b) Merge
    c) Extrapolate
    d) Drop

    return target
    """
    target = []
    resolution = (32 / pow(2, k) )    # cell pixel size

    grid_h = int(new_size[1] // resolution )
    grid_w = int(new_size[0] // resolution )
    lines = [ [(x, y) for (y, x) in zip(line, label['samples']) if y >= 0] for line in label['lines']]
    
    cell_flag = np.zeros((grid_h, grid_w))
    targets = np.zeros((4, grid_h, grid_w))
    cnt = 0
    for line in lines:
        if len(line) == 0:
            continue
        line_lbl = Image.new('L', old_size)
        draw = ImageDraw.Draw(line_lbl)
        draw.line(line, fill='white', width=4)
        line_lbl = np.array( line_lbl.resize(new_size, Image.NEAREST))
        
        for i in range(grid_w):
            window_x1, window_x2 = int(i * resolution), int((i+1) * resolution)
            for j in range(grid_h):
                window_y1, window_y2 = int(j * resolution), int((j+1) * resolution)
                window = line_lbl[window_y1: window_y2, window_x1: window_x2 ]
                args = np.argwhere(window == 255)
                if args.shape[0] <= 2:
                    continue
                
                x1 = np.min( args[..., 1]) / resolution
                x2 = np.max( args[..., 1]) / resolution
                y1 = args[..., 0][ np.argwhere( args[..., 1] == x1 * resolution ) ][0][0] / resolution
                y2 = args[..., 0][ np.argwhere( args[..., 1] == x2 * resolution ) ][0][0] / resolution
                
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # y = mx + b
                m = (y2 - y1) / (x2 - x1 + 1e-6)
                if length <= 0.36:
                    continue
                
                if (-0.1 <= m and m <= 0.1 ) or abs(m) >= 1e6 or dist <= 0.3:
                    y0, y3 = y1, y2
                    x0, x3 = x1, x2

                else:
                    if m < 0:
                        if 0 in (x1, y1) or 1 in (x1, y1):
                            x0, y0 = x1, y1
                        else:
                            x0 = 0
                            y0 = y1 - (x1 - 0) * m
                            if y0 > 1:
                                y0 = 1
                                x0 = x1 - (y1 - 1) / m

                        if 0 in (x2, y2) or 1 in (x2, y2):
                            x3, y3 = x2, y2
                        else:
                            x3 = 1
                            y3 = y2 + (1 - x2) * m
                            if y3 < 0:
                                y3 = 0
                                x3 = x2 + (0 - y2) / m
                    # (m > 0)
                    else:   
                        if 0 in (x1, y1) or 1 in(x1, y1):
                            x0, y0 = x1, y1
                        else:
                            x0 = 0
                            y0 = y2 - x2 * m
                            if y0 < 0:
                                y0 = 0
                                x0 = x2 - y2 / m

                        if 0 in (x2, y2) or 1 in(x2, y2):
                            x3, y3 = x2, y2
                        else:
                            y3 = 1
                            x3 = x2 + (y3 - y2) / m
                            if x3 > 1:
                                x3 = 1
                                y3 = y2 + (x3 - x2) * m
                
                one_grid = np.zeros((4, grid_h, grid_w))
                one_grid[:, j, i] = [x0, y0, x3, y3]

                if cell_flag[j, i] == 0:
                    targets[0: 4, j, i] = [x0, y0, x3, y3]
                    cell_flag[j, i] += 1
                else:
                    kk = int( cell_flag[j, i])
                    if 4 * kk >= targets.shape[0]:
                        targets = np.concatenate((targets, one_grid), axis=0)
                    else:
                        if targets[4 * kk: 4*(kk+1), j, i].sum() == 0:
                            targets[4 * kk: 4*(kk+1), j, i] = [x0, y0, x3, y3]
                        else:
                            targets = np.concatenate((targets, one_grid), axis=0)
                    cell_flag[j, i] += 1

    # print('targets shape: ', targets.shape, 'cell_flag', cell_flag.shape, np.sum(cell_flag))
    return targets