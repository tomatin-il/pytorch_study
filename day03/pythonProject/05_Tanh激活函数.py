import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#创建画布和坐标轴
fig,axes=plt.subplots(1,2)
#函数图象
x=torch.linspace(-20,20,1000)
#输入值x通过Tanh函数转换成激活函值y
y=torch.tanh(x)
#在第一个子图中绘制Tanh激活函数的图像
axes[0].plot(x,y)
axes[0].grid()
axes[0].set_title('Tanh函数图像')
#在第二个图上，绘制Tanh激活函数的导数图像
x=torch.linspace(-20,20,1000,requires_grad=True)
torch.tanh(x).sum().backward()
axes[1].plot(x.detach(),x.grad)
axes[1].set_title('Tanh激活函数导数图像')
axes[1].grid()
plt.show()