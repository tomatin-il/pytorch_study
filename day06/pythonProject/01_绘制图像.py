'''
二值图 :1通道，每个像素点由0，1组成
灰度图：1通道，每个像素点的范围【0，255】
索引图：1通道，每个像素点的范围【0，255】像素点表示颜色表的索引
RGB真彩图：3通道，RED,GREEN,BLUE,红绿蓝
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
import torch

#绘制全黑全白图
def dm01():
    #hwc:高度，宽度，通道
    #全黑
    img1=np.zeros((200,200,3))
    #print(f'img1:{img1}')
    #绘制图片
    plt.imshow(img1)
    plt.show()
    #全白
    img2= torch.full(size=(200, 200, 3),fill_value=255)
    plt.imshow(img2)
    plt.show()

#加载图片
def dm02():
    img1=plt.imread('./data/img.png')
    print(f'img1:{img1}')
    print(f'img1.shape:{img1.shape}')
    plt.imshow(img1)
    plt.show()

if __name__ == '__main__':
    dm02()