import torch
import torch.nn as nn
def dm01():
    y_true=torch.tensor([0,1,0],dtype=torch.float)
    y_pred=torch.tensor(([0.45,0.89,0.22]))
    criterion=nn.BCELoss()
    loss=criterion(y_pred,y_true)
    print(f'损失值：{loss}')
if __name__ == '__main__':
    dm01()