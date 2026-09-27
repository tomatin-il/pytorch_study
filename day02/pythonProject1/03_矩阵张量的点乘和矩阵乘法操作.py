#点乘，两个张量的维度一致，对应元素直接乘法 */mul
#矩阵乘法 @ matmul dot(只针对一维函数)
import torch

def dm01():
    t1=torch.tensor([[1,2,3],[4,5,6]])
    print(f't1:{t1}')
    t2= torch.tensor([[1, 2, 3],[4, 5, 6]])
    print(f't2:{t2}')
    #t3=t1*t2
    t3=t1.mul(t2)
    print(f't3:{t3}')

def dm02():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1:{t1}')
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6],[3,3,3]])
    print(f't2:{t2}')
    t3=t1@t2
    #t3 = t1.matmul(t2)
    print(f't3:{t3}')
    t4=torch.tensor([1,2,3])
    t5 = torch.tensor([2, 2, 3])
    t6=t4.dot(t5)
    print(f't6:{t6}')

if __name__ == '__main__':
    dm01()
   # dm02()