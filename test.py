import torch
import numpy as np
import argparse
import os, time
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR
from hsi_dataset import TrainDataset, ValidDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d
from scipy.signal import savgol_filter
import scipy.io as sio



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
# parser.add_argument('--data_root', type=str, default='/media/jin/b/TG/my_net/train_code/ARAD_1K')
parser.add_argument('--data_root', type=str, default='/media/jin/b/TG/dataset/CAVE_data')
parser.add_argument('--method', type=str, default='CASF_SST_test')
# parser.add_argument('--pretrained_model_path', type=str, default='/media/jin/TGG/my_paper/new/JOCSR_cave_400.pth')
parser.add_argument('--pretrained_model_path', type=str, default='/media/jin/b/TG/my_net/train_code/exp/Our_CAVE_5/loss2025_07_13_14_28_34/net_400epoch.pth')
# parser.add_argument('--pretrained_model_path', type=str, default='/media/jin/b/TG/my_net/train_code/exp/Our_CAVE_3/loss2025_07_09_18_20_40/net_400eopch.pth')
# parser.add_argument('--pretrained_model_path', type=str, default='net_400epoch.pth')
parser.add_argument('--outf', type=str, default='./exp/cave/JOCSR_cave_5/')
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
if torch.cuda.is_available():
    criterion_mrae.cuda()
    criterion_rmse.cuda()


# Validate
# with open(f'{opt.data_root}/split_txt/valid_list+.txt', 'r') as fin:
#     hyper_list = [line.replace('\n', '.png') for line in fin]
# hyper_list.sort()
hyper_list = [str(line) + '.mat' for line in range(32)]
var_name = 'cube'
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    weight_list = []
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(target)
            weight = model.weight
            weight = weight.cpu().numpy().squeeze()
            weight_list.append(weight)
            # output = output.cpu().numpy() * 1.0
            # output = (output - output.min()) / (output.max() - output.min())
            # output = torch.from_numpy(output).cuda()
            # output = input[:, [26, 14, 5], :, :]

            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)

        result = output.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        mat_name = hyper_list[i]
        mat_dir = os.path.join(opt.outf, mat_name)
        # save_matv73(mat_dir, var_name, result)
        output = output.cpu().numpy() * 1.0
        output = (output-output.min())/(output.max()-output.min())
        output = torch.from_numpy(output).cuda()
        print('output', output.shape, mat_dir)
        # img_sample = torch.cat((input.data, target.data, output.data), -1)
        # save_image(img_sample, mat_dir, normalize=True)
        output = np.squeeze(output).cpu().numpy()
        target = np.squeeze(target).cpu().numpy()
        input = np.squeeze(input).cpu().numpy()
        input = np.transpose(np.squeeze(input), [1, 2, 0])
        sio.savemat(mat_dir, {'output':output, 'label':target, 'input':input})
        print("time:", time.time()-start_time)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, weight_list


def filter_curve(x, y):
    coeff = polyfit(x, y, 19)
    f = poly1d(coeff)
    filters_new = f(x)
    return filters_new


if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()
    mrae, rmse, psnr, weight_list = validate(val_loader, model)
    print(f'method:{method}, mrae:{mrae}, rmse:{rmse}, psnr:{psnr}')


    ### plot
    # x = np.arange(400, 701, 10)
    # for i in range(12, 13):
    #     weight = weight_list[i]
    #     plt.plot(x, weight[0, :], color='red')
    #     plt.plot(x, weight[1, :], color='green')
    #     plt.plot(x, weight[2, :], color='blue')
    #     # plt.plot(x, filter_curve(x, weight[0, :]), color='red')
    #     # plt.plot(x, filter_curve(x, weight[1, :]), color='green')
    #     # plt.plot(x, filter_curve(x, weight[2, :]), color='blue')
    #     plt.show()