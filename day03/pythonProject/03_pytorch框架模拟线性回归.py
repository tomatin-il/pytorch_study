#numpy-tensor-tensordataset-dataloader
import torch
from torch.utils.data import TensorDataset #构造数据集对象
from torch.utils.data import DataLoader#数据加载器
from torch import nn #nn模块中有平方损失函数和假设函数
from torch import optim #optim中有优化器函数
from sklearn.datasets import make_regression #线性回归模型数据集
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#创建线性回归样本数据
def creat_dataset():
    x,y,coef=make_regression(
        n_samples=100,#样本数
        n_features=1,#1个特征
        noise=10,
        coef=True,#是否返回系数,默认false返回none
        bias=14.5, #b,coef=w
        random_state=4#随机种子

    )
    #本来xy是numpy，变成张量
    x=torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x,y,coef

#表示模型训练
def train(x,y,coef):
    #1.创建数据集对象
    dataset=TensorDataset(x,y)
    #2.创建数据加载器对象
    #参一：数据集对象 参二：批次大小 参三：是否打乱数据（训练集打乱，测试集不打乱）
    datalaoder=DataLoader(dataset,batch_size=16,shuffle=True)
    #3.创建初始的线性回归模型
    #参一：输入特征维度 参二：输出特征维度
    model=nn.Linear(1,1)
    #4.创建损失函数对象
    criterion=nn.MSELoss()
    #5.创建优化器对象
    #参一：模型参数，参二：学习率
    optimizer=optim.SGD(model.parameters(),lr=0.01)
    #6.具体的训练过程：
    #6.1定义变量，分别表示:训练轮数，每轮的(平均)损失值，训练总损失值，训练的样本数
    epochs,loss_list,total_loss,total_sample=100,[],0.0,0
    for epoch in range(epochs):#eprch的值：0，1到99
        for train_x,train_y in datalaoder:#7批（16，16，16，16，16，16，4）
            #模型预测
            y_pred=model(train_x)
            #计算损失值
            loss=criterion(y_pred,train_y.reshape(-1,1))#-1自动计算
            #计算总损失和样本批次数
            total_loss+=loss.item()
            total_sample+=1
            optimizer.zero_grad()#梯度清零
            loss.backward()#计算梯度
            optimizer.step()#梯度更新
            #把本轮的平均损失值，添加到列表中
        loss_list.append(total_loss/total_sample)
        print(f'轮数：{epoch+1}，平均损失值：{total_loss/total_sample}')
    #7打印最终的训练结果
    print(f'{epochs}轮的平均损失分别为：{loss_list}')
    print(f'权重：{model.weight},偏置：{model.bias}')
    #8绘制损失曲线
    plt.plot(range(epochs),loss_list)
    plt.title('损失曲线变化图')
    plt.grid()
    plt.show()
    #9预测值和真实值的关系
    plt.scatter(x,y)#散点图
    #x100个样本点的特征
    y_pred=torch.tensor(data=[v*model.weight+model.bias for v in x])
    y_true=torch.tensor(data=[v*coef+14.5 for v in x])
    plt.plot(x,y_pred,color='red',label='预测值')
    plt.plot(x, y_true, color='green', label='真实值')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
       x,y,coef= creat_dataset()
      # print(f'x:{x},y:{y},coef:{coef}')
       train(x,y,coef)