
import torch
import torch.nn as nn
#定义函数，演示：多分类交叉熵损失
def dm01():
    #手动创建样本的真实值，y
    y_true=torch.tensor([[0,1,0],[1,0,0]],dtype=torch.float)
    #手动创建样本的预测值，fx
    y_pred=torch.tensor([[0.1,0.8,0.1],[0.7,0.2,0.1]],requires_grad=True,dtype=torch.float)
    #创建多分类交叉熵损失函数
    criterion=nn.CrossEntropyLoss()
    #计算损失值
    loss=criterion(y_pred,y_true)
    print(f'损失值：{loss}')
if __name__ == '__main__':
    dm01()