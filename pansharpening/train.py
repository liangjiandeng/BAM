import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
import scipy.io as sio
from model import BRResNet as NET
import numpy as np
from evaluate import compute_index
import shutil
from torch.utils.tensorboard import SummaryWriter



# ================== Pre-test =================== #
def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy((data['ms'] / 2047.0)).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy((data['pan'] / 2047.0))   # HxW = 256x256

    return lms, ms, pan

def load_gt_compared(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    test_gt = torch.from_numpy(data['gt'] / 2047.0)  # CxHxW = 8x256x256

    return test_gt






# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.001 # 0.0003
epochs = 500
ckpt = 50
batch_size = 32
device=torch.device('cuda:1')

model = NET().to(device)
model.load_state_dict(torch.load('Weight/.pth'))
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))   # optimizer 1


writer = SummaryWriter('train_logs')


def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weight/NET_{}.pth'.format(epoch)  # +500)
    #if not os.path.exists(model_out_path):
    #    os.makedirs(model_out_path)
    torch.save(model.state_dict(), model_out_path)


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader):
    print('Start training...')

    for epoch in range(epochs):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lms, pan = Variable(batch[0], requires_grad=False).to(device), \
                                     Variable(batch[1]).to(device), \
                                     Variable(batch[2]).to(device)
            optimizer.zero_grad()  # fixed
            out = model(pan, lms)

            loss = criterion(out, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch


            loss.backward()  # fixed
            optimizer.step()  # fixed

 #       lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('train/loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, pan= Variable(batch[0], requires_grad=False).to(device), \
                                         Variable(batch[1]).to(device), \
                                         Variable(batch[2]).to(device)

                out = model(pan,lms)


                loss = criterion(out, gt)
                epoch_val_loss.append(loss.item())



        v_loss = np.nanmean(np.array(epoch_val_loss))
        writer.add_scalar('val/loss', v_loss, epoch)
        print('validate loss: {:.7f}'.format(v_loss))

    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":

    train_set = Dataset_Pro('/Data/Machine Learning/Liang-Jian Deng/pansharpening-TNNLS20/02-QB-Data/training_data/train_QB.mat')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('/Data/Machine Learning/Liang-Jian Deng/pansharpening-TNNLS20/02-QB-Data/training_data/validation_QB.mat')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    ###################################################################

    train(training_data_loader, validate_data_loader)
