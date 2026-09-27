import torch

#定义函数，演示：创建线性张量
def dm01():
    #场景一：创建指定类型的线性张量
    t1=torch.arange(0,10,2)
    print(f't1:{t1},type:{type(t1)}')
    print('-'*30)
    #场景二：创建指定范围的线性张量，等差数列
    t2=torch.linspace(1,10,5)
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 30)

#定义函数，演示：创建随机张量
def dm02():
    #场景一:0到1之间随机张量
    #torch.initial_seed()#设置随机种子
    torch.manual_seed(4)#跑两次数值一样

    t1=torch.rand(size=(2,3))
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 30)
    # 场景二:符合正态分布的随机张量
    t2=torch.randn(size=(2,3))
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 30)
    # 场景二:创建随机整数张量
    t3=torch.randint(0,10,size=(2,3))
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 30)
    
#测试函数
if __name__ == '__main__':
    dm02()

