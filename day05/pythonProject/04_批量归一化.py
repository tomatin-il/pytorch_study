#批量归一化：先对数据做标准化（会丢失一些信息），然后在对数据做缩放和平移，在找补回一些信息
import torch
import torch.nn as nn

#定义函数处理二维数据
def dm01():
    #创建图片样本数据
    #1张图片，2个通道，3行4列（像素点）
    input_2d=torch.randn(size=(1,2,3,4))
    print(f'input_2d:{input_2d}')
    #创建批量归一化层（BN层）
    #参一：输入特征数=图片的通道数，参二：噪声值（小常数），参三：动量值 参四：对归一化后的数据进行缩放和平移
    bn2d=nn.BatchNorm2d(num_features=2,eps=1e-5,momentum=0.1,affine=True)
    output_2d=bn2d(input_2d)
    print(f'output_2d:{output_2d}')

def dm02():
    #创建样本数据
    #，2行2列     2条样本，每个样本有2个特征
    input_1d=torch.randn(size=(2,2))
    print(f'input_1d:{input_1d}')
    #创建线性层
    linear1=nn.Linear(2,4)
    l1=linear1(input_1d)
    print(f'l1:{l1}')
    #创建批量归一化层
    bn1d=nn.BatchNorm1d(num_features=4)
    #对线性处理结果l1进行批量归一化处理
    output_1d=bn1d(l1)
    print(f'output_1d:{output_1d}')



if __name__ == '__main__':
    #dm01()
    dm02()

