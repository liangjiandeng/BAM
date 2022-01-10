import torch
from evaluate import compute_index
from scipy import io as sio
from model import BRResNet as NET


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


file_path = "test_data/QB_data2"
test_lms, test_ms, test_pan = load_set(file_path)
test_lms = test_lms.cuda().unsqueeze(dim=0).float()


model=NET().cuda()
model.load_state_dict(torch.load('Weights/DKNET_1000.pth'))

model.eval()
with torch.no_grad():
    output3 = model(test_pan,test_lms)
    result_our = torch.squeeze(output3).permute(1, 2, 0)
    sr = torch.squeeze(output3).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
    result_our = result_our * 2047
    result_our = result_our.type(torch.DoubleTensor).cuda()

    sio.savemat('qb_data2_dknet_q.mat',{'output_our_QB_data2':sr})

    our_SAM, our_ERGAS = compute_index(test_gt, result_our, 4)
    print('our_SAM: {} dmdnet_SAM: 2.9355'.format(our_SAM) ) # print loss for each epoch
    print('our_ERGAS: {} dmdnet_ERGAS:1.8119 '.format(our_ERGAS))