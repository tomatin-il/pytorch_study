'''
张量：存储同一类型元素的容器，且元素值必须是数值
pytorch框架属于最常用的深度学习框架，无论是ANN,还是CNN(图像)，还是RNN
底层在处理数据时，都是使用张量来处理的

张量的基本创建方式：
torch.tensor 根据指定数据创建张量
torch.Torch  根据形状创建张量，其它也可以用来创建指定数据的张量（小写能干的事大写都可以干）
torch.IntTensor、torch.FloatTensor、torch.DoubleTensor创建指定类型的张量
'''

import torch
import numpy as np

#第一步：定义函数，torch.tensor 根据指定数据创建张量
def dm01():
 #场景一：标量 张量
 t1=torch.tensor(10)
 print(f't1:{t1},type:{type(t1)}')
 print('-'*30)
 #场景二：把列表转成张量
 data=[[1,2,3],[4,5,6]]
 t2=torch.tensor(data)
 print(f't2:{t2},type:{type(t2)}')
 print('-' * 30)
 #场景三：numpy ndarray也能转成张量
 data=np.random.randint(0,10,size=(2,3))
 t3=torch.tensor(data,dtype=torch.float)
 print(f't3:{t3},type:{type(t3)}')
 print('-' * 30)
 #场景四：尝试直接创建指定维度 两行三列 的张量
 #t4=torch.tensor(2,3) 报错
 #print(f't4:{t3},type:{type(t4)}')

 #测试代码
if __name__ == '__main__':
   dm01()


#第二步：定义函数，torch.Tensor 根据形状创建张量，其它也可用来创建指定数据的张量
def dm02():
 #场景一：标量 张量
 t1=torch.Tensor(10)
 print(f't1:{t1},type:{type(t1)}')
 print('-'*30)
 #场景二：把列表转成张量
 data=[[1,2,3],[4,5,6]]
 t2=torch.Tensor(data)
 print(f't2:{t2},type:{type(t2)}')
 print('-' * 30)
 #场景三：numpy ndarray也能转成张量
 data=np.random.randint(0,10,size=(2,3))
 t3=torch.Tensor(data)
 print(f't3:{t3},type:{type(t3)}')
 print('-' * 30)
 #场景四：尝试直接创建指定维度 两行三列 的张量
 t4=torch.Tensor(2,3)
 print(f't4:{t3},type:{type(t4)}')

 #测试代码
if __name__ == '__main__':
   dm02()


#第三步：torch.IntTensor、torch.FloatTensor、torch.DoubleTensor创建指定类型的张量
def dm03():
 #场景一：标量 张量
 t1=torch.IntTensor(10)
 print(f't1:{t1},type:{type(t1)}')
 print('-'*30)
 #场景二：把列表转成张量
 data=[[1,2,3],[4,5,6]]
 t2=torch.IntTensor(data)
 print(f't2:{t2},type:{type(t2)}')
 print('-' * 30)
 #场景三：numpy ndarray也能转成张量
 data=np.random.randint(0,10,size=(2,3))
 t3=torch.IntTensor(data)
 print(f't3:{t3},type:{type(t3)}')
 print('-' * 30)
 #场景四：尝试直接创建指定维度 两行三列 的张量
 t4=torch.Tensor(2,3)
 print(f't4:{t3},type:{type(t4)}')
#如果场景不匹配会尝试自动转换类型，
 data=np.random.randint(0,10,size=(2,3))
 t5=torch.FloatTensor(data)
 print(f't5:{t5},type:{type(t5)}')
 print('-' * 30)

 #测试代码
if __name__ == '__main__':
   dm03()
