import torch
from scipy import io as sio
from model import BRResNet as NET
from data import DatasetFromHdf5
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
device=torch.device('cuda:0')


def test():
    C = 31
    test_set = DatasetFromHdf5("/Data/Machine Learning/Jin-Fan Hu/3-multistage/test_harvardv3(with_up)x4.h5")
    num_testing = 10
    sz = 1000
    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=num_testing)
    output = np.zeros([num_testing, sz, sz, C])
    model = NET()
    path_checkpoint = 'NET_1000.pth' # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    model.load_state_dict(checkpoint)  # 加载模型可学习参数
    model = model.to(device)

    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size(), parameters)

    for iteration, batch in enumerate(testing_data_loader, 1):
        GT, HSI, MSI = batch[0], batch[1], batch[2]
        for i in range(num_testing):
            HSIi = HSI[i, :, :, :].to(device)
            HSIi = HSIi[np.newaxis, :, :, :]
            MSIi = MSI[i, :, :, :].to(device)
            MSIi = MSIi[np.newaxis, :, :, :]
            #print(HSIi.shape)
            print(i+1)
            with torch.no_grad():
                outputi = model(MSIi, HSIi)
                output[i, :, :, :] = outputi.permute([0, 2, 3, 1]).cpu().detach().numpy()
                # outputs2[i, :, :, :] = x_stage2.permute([0, 2, 3, 1]).cpu().detach().numpy()
                plt.imshow(MSI[i].permute([1, 2, 0]).squeeze().cpu().detach().numpy())
                plt.axis('off')
                fig = plt.gcf()
                fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.show()
                fig.savefig('S/rgb{}.eps'.format(i+1), format='eps', transparent=True, dpi=300, pad_inches=0,bbox_inches = 'tight')


    sio.savemat('output_dknet_1000.mat', { 'output': output})
    print(output.shape)

if __name__ == '__main__':
    test()