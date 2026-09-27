#sum max min mean 都有dim参数，0为列，1为行
#pow sqrt exp log log2 log10

import torch

t1=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float)

print(f't1:{t1}')
print(t1.sum(dim=0))#按列求和 5，7，9
print(t1.sum(dim=1))#按列求和 6，15
print(t1.sum())#整体求和21
print('-'*30)

print(t1.max(dim=0))#按 列 求最大值
print(t1.max(dim=1))#按 行 求最大值
print(t1.max())     #整体 求最大值
print('-'*30)

print(t1.mean(dim=0))#求平均值
print(t1.mean(dim=1))
print(t1.mean())
print('-'*30)

print(t1.pow(2))#每个数的平方
print(t1**3)#立方
print('-'*30)

print(t1.sqrt())
print('-'*30)

print(t1.exp())#e的n次幂，n就是矩阵中的每个元素
print('-'*30)

print(t1.log())
print(t1.log2())
print(t1.log10())
print('-'*30)
