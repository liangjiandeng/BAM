import torch
from scipy import io as sio
from model import BRResNet as NET
from data import DatasetFromHdf5
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
device=torch.device('cuda:1')


def test():
    C = 31


    test_set = DatasetFromHdf5("/Data/Machine Learning/Zi-Rong Jin/LAConv/hyp/h/test_data/test_cave4_11.h5")
    num_testing = 11
    sz = 512
    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=num_testing)
    output = np.zeros([num_testing, sz, sz, C])
    model = NET()
    path_checkpoint = 'Weights/NET_1000.pth' # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    model.load_state_dict(checkpoint)  # 加载模型可学习参数
    model = model.to(device)

    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size(), parameters)

    for iteration, batch in enumerate(testing_data_loader, 1):
        GT, HSI, MSI = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        for i in range(num_testing):
            HSIi = HSI[i, :, :, :]
            HSIi = HSIi[np.newaxis, :, :, :]
            MSIi = MSI[i, :, :, :]
            MSIi = MSIi[np.newaxis, :, :, :]
            print(HSIi.shape)
            with torch.no_grad():
                outputi = model(MSIi, HSIi)
                output[i, :, :, :] = outputi.permute([0, 2, 3, 1]).cpu().detach().numpy()
                # plt.imshow(MSI[i].permute([1, 2, 0]).squeeze().cpu().detach().numpy())
                # plt.axis('off')
                # fig = plt.gcf()
                # fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                # plt.margins(0, 0)
                # plt.show()
                # fig.savefig('S/rgb{}.eps'.format(i+1), format='eps', transparent=True, dpi=300, pad_inches=0,bbox_inches = 'tight')
                #outputs2[i, :, :, :] = x_stage2.permute([0, 2, 3, 1]).cpu().detach().numpy()


    sio.savemat('output_dknet_1000.mat', { 'output': output})

if __name__ == '__main__':
    test()