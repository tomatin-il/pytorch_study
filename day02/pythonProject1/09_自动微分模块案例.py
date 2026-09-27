#w新=w旧-学习率*梯度
import torch
w=torch.tensor(10,requires_grad=True,dtype=torch.float32)
loss=w**2+20
#利用梯度下降法，循环迭代100，求最优解
print(f'开始权重初始值；{w},(0.01*w.grad)：无，loss{loss}')
#正向计算（前向传播）
for i in range(1,101):
    loss=w**2+20
    #梯度清0，因为默认会累加
    #至此（第一次的时候），还没有计算梯度，所以w.grad=none,要做非空判断
    if w.grad is not None:
        w.grad.zero_()
    loss.sum().backward()
    print(f'梯度值为：{w.grad}')
    w.data=w.data-0.01*w.grad
    print(f'第{i}次,权重初始值：{w}，(0.01*w.grad):{0.01*w.grad},loss:{loss}')

