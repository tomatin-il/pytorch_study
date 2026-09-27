import torch
import torch.nn as nn
#定义函数，演示单通道池化
def dm01():
    #创建1个1通道3*3的二维矩阵
    inputs=torch.tensor(
        [
            [
                [0,1,2],
                [3,4,5],
                [6,7,8]
            ]
        ]
    )
   # print(f'inputs:{inputs},shape:{inputs.shape}')
   #2创建最大池化层
   #参一：池化核大小2*2，参二：步长 窗口每次向右下滑动1格，参三：填充外围不补0
    pool1=nn.MaxPool2d(2,1,0)
    outputs=pool1(inputs)
    print(f'outputs:{outputs},shape:{outputs.shape}')
   #3创建平均池化层
    pool2=nn.AvgPool2d(2,1,0)
    outputs = pool2(inputs)
    print(f'outputs:{outputs},shape:{outputs.shape}')


#定义函数，演示多通道池化
def dm02():
    #创建1个3通道3*3的二维矩阵
    inputs=torch.tensor(
        [
            [
                [0,1,2],
                [3,4,5],
                [6,7,8]
            ],
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]
            ],
            [
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99]
            ]
        ]
    )
   # print(f'inputs:{inputs},shape:{inputs.shape}')
   #2创建最大池化层
   #参一：池化核大小2*2，参二：步长 窗口每次向右下滑动1格，参三：填充外围不补0
    pool1=nn.MaxPool2d(2,1,0)
    outputs=pool1(inputs)
    print(f'outputs:{outputs},shape:{outputs.shape}')
   #3创建平均池化层
    pool2=nn.AvgPool2d(2,1,0)
    outputs = pool2(inputs)
    print(f'outputs:{outputs},shape:{outputs.shape}')
#测试
if __name__ == '__main__':
    dm02()