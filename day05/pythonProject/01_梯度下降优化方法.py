import torch
import torch.nn as nn
import torch.optim as optim
#动量法（momentum）
def dm01_momentum():
    w=torch.tensor(1.,requires_grad=True,dtype=torch.float32)
    #损失函数
    criterion=(w**2/2)
    #参一：待优化的参数列表，参二：学习率，参三：动量参数
    #优化器：基于SGD加入参数momentum，就是动量法
    optimizer=optim.SGD(params=[w],lr=0.01,momentum=0.9)
    #计算梯度值：梯度清0+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    #第二次更新权重参数
    criterion = (w ** 2 / 2)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

#自适应学习率AdaGrad
def dm02_adagrad():
    w=torch.tensor(1.,requires_grad=True,dtype=torch.float32)
    #损失函数
    criterion=(w**2/2)
    #参一：待优化的参数列表，参二：学习率，参三：动量参数
    optimizer=optim.Adagrad(params=[w],lr=0.01)
    #计算梯度值：梯度清0+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    #第二次更新权重参数
    criterion = (w ** 2 / 2)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

#自适应学习率RMSprop
def dm03_rmsprop():
    w=torch.tensor(1.,requires_grad=True,dtype=torch.float32)
    #损失函数
    criterion=(w**2/2)
    #参一：待优化的参数列表，参二：学习率，参三：动量参数
    optimizer=optim.RMSprop(params=[w],lr=0.01,alpha=0.9)
    #计算梯度值：梯度清0+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    #第二次更新权重参数
    criterion = (w ** 2 / 2)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')


#自适应矩估计adam
def dm04_adam():
    w=torch.tensor(1.,requires_grad=True,dtype=torch.float32)
    #损失函数
    criterion=(w**2/2)
    #参一：待优化的参数列表，参二：学习率，参三：动量参数
    optimizer=optim.Adam(params=[w],lr=0.01,betas=(0.9,0.999))
    #计算梯度值：梯度清0+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    #第二次更新权重参数
    criterion = (w ** 2 / 2)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')


if __name__ == '__main__':
    #dm01_momentum()
    #dm02_adagrad()
     #dm03_rmsprop()
     dm04_adam()