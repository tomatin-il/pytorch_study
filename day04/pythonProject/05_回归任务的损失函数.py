import torch
import torch.nn as nn
def dm01():
    y_true=torch.tensor([2,2,2],dtype=torch.float)
    y_pred=torch.tensor([1,1,1.9],requires_grad=True)
    #MAE损失函数
    criterion=nn.L1Loss()
    loss=criterion(y_pred,y_true)
    print(f'MAE:{loss}')

def dm02():
    y_true=torch.tensor([2,2,2],dtype=torch.float)
    y_pred=torch.tensor([1,1,1.9],requires_grad=True)
    #MSE损失函数
    criterion=nn.MSELoss()
    loss=criterion(y_pred,y_true)
    print(f'MSE:{loss}')

def dm03():
    y_true=torch.tensor([2,2,2],dtype=torch.float)
    y_pred=torch.tensor([1,1,1.9],requires_grad=True)
    #MSE损失函数
    criterion=nn.SmoothL1Loss()
    loss=criterion(y_pred,y_true)
    print(f'Smooth L1:{loss}')

if __name__ == '__main__':
    dm03()



