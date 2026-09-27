import torch
import numpy as np
#场景一：张量转换为numpy nd数组对象
def dm01():
    t1=torch.tensor([1,2,3,4,5])
    print(f't1:{t1},type:{type(t1)}')
   # n1=t1.numpy() #共享内存
    n1=t1.numpy().copy()#不共享内存
    print(f'n1:{n1},type:{type(n1)}')
    #演示上述方式共享内存
    n1[0]=100
    print(f'n1:{n1}')
    print(f't1:{t1}')

#场景二：numpy nd数组转换为 张量
def dm02():
    n1=np.array([11,22,33])
    print(f'n1:{n1},type:{type(n1)}')
    t1=torch.from_numpy(n1)#共享内存
    print(f't1:{t1},type:{type(t1)}')
    t2=t1.type(torch.float32)
    print(f't2:{t2},type:{type(t2)}')
    t3=torch.tensor(n1)#不共享内存
    print(f't3:{t3},type:{type(t3)}')
    n1[0]=100
    print(f't1:{t1}')
    print(f't3:{t3}')

#场景三：从标量张量（只有一个值）中提取其内存
def dm03():
    #t1=torch.tensor(100.3)#只能提取一个数，只能是数值
    t1=torch.tensor([100,])
    print(f't1:{t1},type:{type(t1)}')
    a=t1.item()
    print(f'a:{a},type:{type(a)}')

if __name__ == '__main__':
    dm03()