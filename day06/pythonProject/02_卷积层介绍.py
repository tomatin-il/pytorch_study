import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def dm01():
    img=plt.imread('./data/img.png')
    #print(f'img:{img},shape:{img.shape}')

    #把HWC变成CHW,img-张量-维度转换
    img2=torch.tensor(img,dtype=torch.float32)
    img2=img2.permute(2,0,1)

    #CHW-(1,C,H,W)
    img3=img2.unsqueeze(dim=0)

    #创建卷积层对象
    #参一：输入图像的通道数，参二：输出图像的通道数（几个特征图）参三：卷积核的大小 参四：步长，参五：填充
    conv=nn.Conv2d(4,4,3,2,0)
    conv_img=conv(img3)
    #一张4通道，4是out_channels卷积核的个数
    print(f'conv_img;{conv_img},shape:{conv_img.shape}')

    img4=conv_img[0]
    img5 = img4.permute(1, 2, 0)
    #可视化第一个通道的特征图
    feature1=img5[:,:,0].detach().numpy()
    plt.imshow(feature1)
    plt.show()
if __name__ == '__main__':
    dm01()