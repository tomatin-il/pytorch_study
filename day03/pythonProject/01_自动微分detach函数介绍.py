#一个张量一旦设置了自动微分，这个张量就不能直接转成numpy的ndarray对象了，需要通过detach（）
import torch
import numpy as np
#定义张量
#t1=torch.tensor([10,20],dtype=torch.float) 可以转换
t1=torch.tensor([10,20],requires_grad=True,dtype=torch.float)
print(f't1:{t1},type:{type(t1)}')
#把上述张量转成numpy对象
#n1=t1.numpy()
#print(f'n1:{n1},type:{type(n1)}')
t2=t1.detach()
print(f't2:{t2},type:{type(t2)}')
#测试t1和t2是共享同一份空间
t1.data[0]=100
print(f't1:{t1},type:{type(t1)}')
print(f't2:{t2},type:{type(t2)}')
#查看t1和t2谁可以自动微分
print(f't1:{t1.requires_grad},t2:{t2.requires_grad}')
#把t2转成numpy对象
n1=t2.numpy()
print(f'n1:{n1},type:{type(n1)}')
n2=t1.detach().numpy()
print(f'n2:{n2},type:{type(n2)}')
