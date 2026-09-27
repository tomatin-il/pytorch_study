import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import optim
import matplotlib.pyplot as plt
#等间隔学习率衰减
def dm01():
    #初始的学习率，训练的轮数，每轮训练的批次数
    lr,epochs,iteration=0.1,200,10
    #创建数据集，y_true真实值,输入特征x,权重参数w
    y_true=torch.tensor([0])
    x=torch.tensor(1,dtype=torch.float)
    #权重参数w，需要自动微分
    w=torch.tensor(1,requires_grad=True,dtype=torch.float32)
    #创建优化器参数，动量法，加速模型收敛，减小震荡
    optimizer=optim.SGD([w],lr=lr,momentum=0.9)
    #创建学习率衰减对象
    #参一：优化器对象 参二:间隔的轮数（多少轮调整一次学习率），参三：学习率衰减系数
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
    #创建两个列表，分别表示：训练轮数，每轮训练用的学习率
    lr_list,epoch_list=[],[]
    #循环遍历训练轮数，进行具体的训练
    for epoch in range(epochs): #epoch:0-199
      #获取当前轮数和学习率，并保存到列表中
      epoch_list.append(epoch)
      lr_list.append(scheduler.get_last_lr())
      for batch in range(iteration):
          y_pred=w*x
          loss=(y_pred-y_true)**2
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      #更新学习率
      scheduler.step()
    print(f'lr_list:{lr_list}')
    plt.plot(epoch_list,lr_list,label="Learning Rate")
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend() #图例
    plt.show()

#指定间隔学习率衰减
def dm02():
    #初始的学习率，训练的轮数，每轮训练的批次数
    lr,epochs,iteration=0.1,200,10
    #创建数据集，y_true真实值,输入特征x,权重参数w
    y_true=torch.tensor([0])
    x=torch.tensor(1,dtype=torch.float)
    #权重参数w，需要自动微分
    w=torch.tensor(1,requires_grad=True,dtype=torch.float32)
    #创建优化器参数，动量法，加速模型收敛，减小震荡
    optimizer=optim.SGD([w],lr=lr,momentum=0.9)
    #创建学习率衰减对象
    #参一：优化器对象 参二:间隔的轮数（多少轮调整一次学习率），参三：学习率衰减系数
    #定义变量，记录要修改学习率的轮数
    milestones=[10,20,50,125,145,160]
    scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.5)
    #创建两个列表，分别表示：训练轮数，每轮训练用的学习率
    lr_list,epoch_list=[],[]
    #循环遍历训练轮数，进行具体的训练
    for epoch in range(epochs): #epoch:0-199
      #获取当前轮数和学习率，并保存到列表中
      epoch_list.append(epoch)
      lr_list.append(scheduler.get_last_lr())
      for batch in range(iteration):
          y_pred=w*x
          loss=(y_pred-y_true)**2
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      #更新学习率
      scheduler.step()
    print(f'lr_list:{lr_list}')
    plt.plot(epoch_list,lr_list,label="Learning Rate")
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend() #图例
    plt.show()
#指数学习率衰减
def dm03():
    #初始的学习率，训练的轮数，每轮训练的批次数
    lr,epochs,iteration=0.1,200,10
    #创建数据集，y_true真实值,输入特征x,权重参数w
    y_true=torch.tensor([0])
    x=torch.tensor(1,dtype=torch.float)
    #权重参数w，需要自动微分
    w=torch.tensor(1,requires_grad=True,dtype=torch.float32)
    #创建优化器参数，动量法，加速模型收敛，减小震荡
    optimizer=optim.SGD([w],lr=lr,momentum=0.9)
    #创建学习率衰减对象
    #参一：优化器对象 参二:间隔的轮数（多少轮调整一次学习率），参三：学习率衰减系数
    #定义变量，记录要修改学习率的轮数

    scheduler=optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
    #创建两个列表，分别表示：训练轮数，每轮训练用的学习率
    lr_list,epoch_list=[],[]
    #循环遍历训练轮数，进行具体的训练
    for epoch in range(epochs): #epoch:0-199
      #获取当前轮数和学习率，并保存到列表中
      epoch_list.append(epoch)
      lr_list.append(scheduler.get_last_lr())
      for batch in range(iteration):
          y_pred=w*x
          loss=(y_pred-y_true)**2
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      #更新学习率
      scheduler.step()
    print(f'lr_list:{lr_list}')
    plt.plot(epoch_list,lr_list,label="Learning Rate")
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend() #图例
    plt.show()





#测试
if __name__ == '__main__':
    #dm01()
    #dm02()
    dm03()