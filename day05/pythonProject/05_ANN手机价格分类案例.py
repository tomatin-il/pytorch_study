#基于手机的20列特征，预测手机的价格特征4个区间
#构建数据集 搭建神经网络 模型训练 模型测试
#优化思路 sgd-adam 学习率0.001-0.0001 对数据进行标准化 增加网络的深度，每层的神经元个数 调整训练的轮数
import torch#pytorch框架，封装了张量的各种操作
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset#数据集对象，数据 →tensor→数据集→数据加载器
from torch.utils.data import DataLoader#数据加载器
import torch.nn as nn #neural network封装了神经网络的各种操作
import torch.optim as optim #优化器
from sklearn.model_selection import train_test_split#切分训练集和测试集
import matplotlib.pyplot as plt#绘图
import numpy as np#数组矩阵操作
import pandas as pd#数据处理
import time#时间模块
from torchsummary import summary

#todo:1定义函数，构建数据集
def  create_dataset():
    #1 加载csv文件数据集
    data=pd.read_csv('./data/手机价格预测.csv')
    #print(f'data:{data.head()}')
    #print(f'data:{data.shape}')#2000行每列21列

    #2 获取x特征列，和y标签列
    x,y=data.iloc[:,:-1],data.iloc[:,-1]#x取所有行，列从第0列到倒数第二列  y取所有行，只取最后一列

    #3 把特征列转成浮点型
    x=x.astype(np.float32)

    #4 切分训练集和测试集
    #参一：特征 参二：标签 参三：测试集所占的比例 参4：随机种子  参5：样本的分布  抽取数据集的时候参考y的比例
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4,stratify=y)

    #优化1:数据标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)



    #5把数据集封装成张量数据集
    train_dataset=TensorDataset(torch.from_numpy(x_train),torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.from_numpy(x_test),torch.tensor(y_test.values))

    #返回结果                           20充当输入特征数    4输出标签数
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))

#todo:2搭建神经网络
class PhonePriceModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        #1.1初始化父类成员
        super().__init__()
        #1.2搭建神经网络
        self.linear1=nn.Linear(input_dim,128)#2688=128*21
        self.linear2 = nn.Linear(128, 256)#256*129=33024
        self.output = nn.Linear(256, 4)#257*4=1028
        #2.定义前向传播方法forward()
    def forward(self,x):
        #2.1隐藏层1：加权求和+激活函数（relu）
        x=torch.relu(self.linear1(x))
        #2.2隐藏层2：加权求和+激活函数（relu）
        x=torch.relu(self.linear2(x))
        x=self.output(x)
        return x


#todo:3模型训练
def train(train_dataset,input_dim,output_dim):
    #创建数据加载器 数据-张量-数据集-数据加载器
    #参一：数据集对象1600，参二：每批次的数据条数 参三：是否打乱数据集 训练集打乱，测试集不打乱
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    #2创建神经网络模型
    model=PhonePriceModel(input_dim,output_dim)
    #3定义损失函数，因为是多分类，这里用的是：多分类交叉熵损失函数
    criterion=nn.CrossEntropyLoss()
    #4创建优化器对象
    optimizer=optim.Adam(model.parameters(),lr=0.0001)
    #5.模型训练
    #5.1记录训练的轮数
    epochs=100
    #5.2开始每轮的训练
    for epoch in range(epochs):
        total_loss,batch_num=0.0,0
        start=time.time()
        for x,y in train_loader:
            model.train()#训练模式
            y_pred=model(x)
            loss=criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()#把本轮的每批次16条的平均损失累积起来
            batch_num +=1
        #至此，本轮训练结束，打印训练信息
        print(f'epoch:{epoch+1},loss:{total_loss/batch_num:4f},time:{time.time()-start:2f}s')

    #这里多轮训练结束，保存模型参数
    #参1：模型对象的参数（权重矩阵，偏置矩阵）参2：模型保存的文件名
    torch.save(model.state_dict(),'./model/phone.pth')

#todo:4模型测试
def evaluate(test_dataset,input_dim,output_dim):
    #1.创建神经网络分类对象
    model=PhonePriceModel(input_dim,output_dim)
    #2.加载参数模型
    model.load_state_dict(torch.load('./model/phone.pth'))
    #3.创建测试集的数据加载器对象
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    #4.定义变量，记录预测正确的样本数
    correct=0
    #5.从数据加载器中获取到每批次的数据
    for x,y in test_loader:
        model.eval()
        y_pred=model(x)
        #argmax获取最大值对应的下标，就是类别
        y_pred=torch.argmax(y_pred,dim=1)
        #print(f'y_pred:{y_pred}')
        #print(f'y:{y}')
        #预测正确的样本个数
        #print(y_pred==y)
        #print((y_pred==y).sum())
        correct+=(y_pred==y).sum()

    print(f'准确率:{correct/len(test_dataset):.4f}')

#todo:5测试
if __name__ == '__main__':
    #1.准备数据集
    train_dataset,test_dataset,input_dim,output_dim=create_dataset()
    #2.构建神经网络模型
   # model=PhonePriceModel(input_dim,output_dim)
    #参一：模型对象 参二：输入数据的形状，每批16条，每条20条特征
    #summary(model,input_size=(16,input_dim))
    train(train_dataset,input_dim,output_dim)
    evaluate(test_dataset,input_dim,output_dim)