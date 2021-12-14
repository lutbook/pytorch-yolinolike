import os, time, functools, argparse, torch, cv2
import numpy as np
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from yolino_net import net
from yolino_dataset import YolinoDataset
from functions import YolinoLoss, iou_v2, inference

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("-exp", "--exp_name", type=str, help='expirement name', default='exp0')
parser.add_argument("-ds", "--dataset_dir", type=str, help='dataset directory', default='data')
parser.add_argument("-ne", "--num_epochs", type=int, help='number of training epochs', default=100)
parser.add_argument("-bs", "--batch_size", type=int, help='batch size', default=2)
parser.add_argument("-pd", "--pred_dir", type=str, help='prediction dir', default=None)
parser.add_argument("-m", "--model", type=str, help='model', default='yolinonet')
parser.add_argument("-imh", "--image_height", type=int, help='image size', default=448)
parser.add_argument("-imw", "--image_width", type=int, help='image size', default=960)
parser.add_argument("-ct", "--conf_thresh", type=int, help='confidence threshold', default=0.5)

args = parser.parse_args()
ROOT_DIR = os.path.dirname(os.getcwd())
os.chdir(ROOT_DIR)
EXP_NAME = args.exp_name
MODEL_DIR = 'pytorch_yolino'
DATASET_DIR = os.path.join(ROOT_DIR, args.dataset_dir) #args.dataset_dir
PRED_DIR = os.path.join( args.pred_dir) if not args.pred_dir==None else None
MODEL = args.model
MODEL_NAME = os.path.join(EXP_NAME, 'weights', '{}_epoch_%d.pth'.format(MODEL))
BATCH_SIZE = args.batch_size # default=2
EPOCHS = args.num_epochs
RESUME = False
SAVING_STEP = 5
K = 1
P = 8
IMG_HEIGHT = args.image_height
IMG_WIDTH = args.image_width
CONF_THRESH = args.conf_thresh / 10 if type(args.conf_thresh) == int else args.conf_thresh

print('\n', '*   -----   ' * 7, '*')
# Experiment directory check
if not os.path.isdir(EXP_NAME):
    os.mkdir(EXP_NAME)
    os.mkdir(os.path.join(EXP_NAME, 'weights'))
    os.mkdir(os.path.join(EXP_NAME, 'tb_log'))
    print("Experiment : '{}' has begin.".format(EXP_NAME))
else:
    # if not PRED_DIR:
    RESUME = True

# Tensorboard log
writer = SummaryWriter(os.path.join(EXP_NAME, 'tb_log'))

def running(model, device, dataset_dir, train, resume, epochs, input_size, batch_size, p, k):
    print('running process ..')
    train_loss = []
    val_loss = []
    iou = 0
    start_epoch = 0
    best_miou = 0
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = YolinoLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # if RESUME or not PRED_DIR==None:
    if RESUME:
        print("Continue from experiment: '{}'".format(EXP_NAME))
        try:
            os.remove(os.path.join(EXP_NAME, 'weights', '.DS_Store'))
        except:
            pass
        if PRED_DIR:
                s = 'inference'
                if not os.path.isdir( os.path.join(EXP_NAME, s) ):
                    os.mkdir(os.path.join(EXP_NAME, s))
                    print("Prediction result will be saved in '{}'\n".format(os.path.join(EXP_NAME, s)))
                checkpoint = torch.load(MODEL_NAME % 0)
                f = open(os.path.join(EXP_NAME, s, 'inference result.txt'), 'w+')
                print("\t(mIoU: {:.4f} model loaded: '{}')\n\n".format(checkpoint['mIoU'], MODEL_NAME % 0))
                f.writelines("\nmIoU: {:.4f} model loaded: '{}'\n\n".format(checkpoint['mIoU'], MODEL_NAME % 0))
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
        else:        
            chkpts_list = os.listdir(os.path.join(EXP_NAME, 'weights'))
            if len(chkpts_list) != 0:
                latest_epoch_saved = np.amax(np.array([int( x.split('.')[0].split('_')[-1] ) for x in chkpts_list]))
                checkpoint = torch.load(MODEL_NAME % latest_epoch_saved)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                best_miou = checkpoint['mIoU']
                print('\tresuming from:', os.path.join(EXP_NAME, 'weights', '{}_epoch_%d.pth'.format(MODEL) % latest_epoch_saved),'\n')
                if start_epoch >= EPOCHS:
                    print('')
                    print("Training epoch is {}, but loaded epoch is {}.".format(EPOCHS, start_epoch))
                    print("Try again with higher epoch number.\n")
                    exit(0)
    
    if not PRED_DIR:
        print('Training ...\n')
        image_transform = transforms.Compose([
            transforms.Resize(input_size, InterpolationMode.NEAREST ),
            transforms.ColorJitter(),
            transforms.ToTensor()
        ])
        train_dataset = YolinoDataset(dataset_dir=dataset_dir, image_transform=image_transform, phase='train', p=p, k=k)
        val_dataset = YolinoDataset(dataset_dir=dataset_dir, image_transform=image_transform, phase='val', p=p, k=k)

        train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                        batch_size, 
                                                        pin_memory=True,
                                                        shuffle=True,
                                                        num_workers=0)
        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size,
                                                      pin_memory=True,
                                                      shuffle=True,
                                                      num_workers=0)

        # training
        for epoch in range(start_epoch, epochs):
            model.train()
            train_epoch_loss = 0
            start_time = time.time()
            for i, (inputs, target) in enumerate(train_data_loader):
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                pred = model(inputs)
                loss = criterion(target, pred)
                train_epoch_loss += loss.item()
                # print('training iter {}: loss: {} '.format(i, loss.item())) ############################
                if device.type == 'cpu':
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            train_loss.append( train_epoch_loss / len(train_data_loader) )

            # validation
            model.eval()
            val_epoch_loss = 0
            _iou = 0
            with torch.no_grad():
                for i, (inputs, target) in enumerate(val_data_loader):
                    inputs = inputs.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    output = model(inputs)
                    loss = criterion(target, output)
                    # print('val iter {}: loss: {} '.format(i, loss.item())) ############################
                    val_epoch_loss += loss.item()
                    _iou += iou_v2(target=target, pred=output, img_size=(IMG_WIDTH, IMG_HEIGHT), k=K, conf_thresh=CONF_THRESH)
                val_loss.append( val_epoch_loss / len(val_data_loader) )
                iou = _iou / len(val_data_loader)

            scheduler.step()
            end_time = time.time()
            print('-- Epoch {} -- train_loss: {:.4f}, val_los: {:.4f}   --mIoU: {:.4f}     ({:.4f} mins)'.format(epoch,
                                                                                                                 train_loss[epoch - start_epoch],
                                                                                                                 val_loss[epoch - start_epoch],
                                                                                                                 iou,
                                                                                                                 (end_time - start_time)/60))
            
            writer.add_scalars('Loss', {'train loss': train_loss[epoch - start_epoch],
                                        'val loss': val_loss[epoch - start_epoch],
                                        'mIoU': iou}, epoch)

            if epoch % SAVING_STEP == (SAVING_STEP - 1):
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mIoU': iou.mean()
                }, MODEL_NAME % (epoch + 1))

            if best_miou <= iou:
                best_miou = iou
                print("\t\t\t\t\t\t\tBest mIoU model: {}: {:.4f} mIoU".format(MODEL_NAME % 0, best_miou))
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mIoU': best_miou
                }, MODEL_NAME % 0)
        
      
    # inference on data
    else:
        print('Inference..')
        f.writelines('\nInference file name, size, fps\n')
        try:
            os.remove(os.path.join(PRED_DIR, '.DS_Store'))
        except:
            pass
    
        for file_name in sorted(os.listdir(PRED_DIR)):
            if not os.path.isfile(os.path.join(PRED_DIR, file_name)):
                continue
            if file_name[0] == '.':
                continue
        
            # inference on '*.mp4' video files
            elif file_name.split('.')[1] == 'mp4':
                file_path = os.path.join(PRED_DIR, file_name)
                out_file_path = os.path.join(EXP_NAME, s, file_name)
                cap = cv2.VideoCapture(file_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                out_video = cv2.VideoWriter(out_file_path.split('.')[0]+'_masked.mp4', 
                                            cv2.VideoWriter_fourcc(*'mp4v'), 
                                            fps, 
                                            (frame_width, frame_height) )
                mask_video = cv2.VideoWriter(out_file_path.split('.')[0]+'_mask_only.mp4', 
                                            cv2.VideoWriter_fourcc(*'mp4v'), 
                                            fps, 
                                            (frame_width, frame_height) )
                duration = 0
                frm_cntr = 0
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)

                for frame in frames:
                    frm_cntr += 1
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    output = img
                    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                    start_time = time.time()
                    image_tensor = transforms.ToTensor()(img)
                    mask = Image.new("RGB", img.size)

                    _pred = model(image_tensor.unsqueeze(0).to(device, non_blocking=True))
                    _pred_img = inference(_pred, img_size=img.size, k=K, conf_thresh=CONF_THRESH)
                    Image.Image.paste(mask, _pred_img)

                    mask = mask.resize(output.size, Image.NEAREST)
                    output = Image.composite(mask, output , mask.convert('L'))
                    out_video.write(np.array(output)[:, :, :: -1] )
                    mask_video.write(np.array(mask)[:, :, :: -1] )
                    end_time = time.time()
                    duration += end_time - start_time
                    print("\t\tvideo frame processing: {}/{}".format(frm_cntr, n_frames))

                cap.release()
                out_video.release()
                mask_video.release()
                str = '{} : size= {} original fps: {:.4f}, model fps: {:.4f}'.format(file_name, 
                                                                                     img.size, 
                                                                                     fps,
                                                                                     n_frames / duration * 1.0 )
                print(str)
                f.writelines('\n\t' + str)

            # inference on '*.png' '*.jpg' image files
            elif file_name.split('.')[1] == 'png' or file_name.split('.')[1] == 'jpg':
                file_path = os.path.join(PRED_DIR, file_name)
                out_file_path = os.path.join(EXP_NAME, s, file_name)
                img = Image.open(file_path).convert('RGB')
                start_time = time.time()
                blend_output = img
                masked_output = img
                img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                image_tensor = transforms.ToTensor()(img)
                mask = Image.new("RGB", img.size)

                _pred = model(image_tensor.unsqueeze(0).to(device, non_blocking=True))
                _pred_img = inference(_pred, img_size=img.size, k=K, conf_thresh=CONF_THRESH)
                Image.Image.paste(mask, _pred_img)
        
                mask = mask.resize(blend_output.size, Image.NEAREST)
                blend_output = Image.composite(mask, blend_output , mask.convert('L'))
                masked_output = mask
                end_time = time.time()
                blend_output.save(out_file_path.split('.')[0]+'_blend.png')
                masked_output.save(out_file_path.split('.')[0]+'_mask.png')
                str = '{}: size={}, fps:{:.4f}'.format(file_name, 
                                                       img.size,
                                                       1/(end_time-start_time))
                print(str)
                f.writelines('\n\t' + str)

            # other files are not compatible 
            else:
                print("Your file: ", file_name ,"\n\tChoose '.png','.jpg' image file or '.mp4' video file. \n")
                continue
        f.close()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device.type={}, device.index={}\n'.format( device.type, device.index))

    if MODEL == 'yolinonet':
        model = net(p=P)
    else:
        print('No {} model defined!'.format(MODEL))
        exit(0)
    
    model.to(device)
    running(model=model, device=device, dataset_dir=DATASET_DIR, train=PRED_DIR, resume=RESUME, epochs=EPOCHS,
            input_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, p=P, k=K)
    writer.close()

if __name__ == '__main__':
    main()