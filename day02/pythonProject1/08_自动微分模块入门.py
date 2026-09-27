#只有标量张量才能求导
import torch
#定义变量，记录初始的权重
#参一：初始值，参二：是否自动微分，参三：数据类型
w=torch.tensor(10,requires_grad=True,dtype=torch.float)
#定义loss变量表示损失函数
loss=2*w**2
#print(f'梯度函数类型：{type(loss.grad_fn)}')
#print((loss.sum()))#代入10
#计算梯度，梯度是损失函数的导数，计算完毕后会记录到 w.grad属性中
loss.sum().backward()#保证loss是标量
#代入权重更新公式：W新=W旧-学习率*梯度
w.data=w.data -0.01*w.grad
print(f'更新后的权重：{w}')

