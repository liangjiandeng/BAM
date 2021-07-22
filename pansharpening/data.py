import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np




class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047
        self.gt = torch.from_numpy(gt1).permute(0,3,1,2)  # NxCxHxW:

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047
        self.lms = torch.from_numpy(lms1).permute(0,3,1,2)

        ms1 = data["ms"][...]  # NxCxHxW
        ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047  # NxHxWxC
        ms1_tmp = get_edge(ms1)  # NxHxWxC
        self.ms_hp = torch.from_numpy(ms1_tmp).permute(0, 3, 1, 2) # NxCxHxW:


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047  # Nx1xHxW
        self.pan = torch.from_numpy(pan1).unsqueeze(1)  # Nx1xHxW:
        print(self.lms.shape,self.pan.shape)

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.ms_hp[index, :, :, :].float(), \
               self.pan[index, :, :, :].float() # Nx1xHxW:
            #####必要函数
    def __len__(self):
        return self.gt.shape[0]
