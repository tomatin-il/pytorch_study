import torch

t1=torch.ones(2,3)#创建2行3列全1张量
print(f't1:{t1},type:{type(t1)}')
print('-'*30)
t2=torch.tensor([[1,2],[3,4],[5,6]])
t3=torch.ones_like(t2)#基于t2的形状创建全1张量
print(f't3:{t3},type:{type(t3)}')
print('-'*30)


t1=torch.zeros(2,3)#创建2行3列全0张量
print(f't1:{t1},type:{type(t1)}')
print('-'*30)
t2=torch.tensor([[1,2],[3,4],[5,6]])
t3=torch.zeros_like(t2)#基于t2的形状创建全0张量
print(f't3:{t3},type:{type(t3)}')
print('-'*30)

t1=torch.full(size=(2,3),fill_value=255)#创建2行3列全255张量，255是白色
print(f't1:{t1},type:{type(t1)}')
print('-'*30)
t2=torch.tensor([[1,2],[3,4],[5,6]])
t3=torch.full_like(t2,fill_value=255)#基于t2的形状创建全255张量
print(f't3:{t3},type:{type(t3)}')
print('-'*30)

