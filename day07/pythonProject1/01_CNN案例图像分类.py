# 导包
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

#每批次样本数
BATCH_SIZE=8
#准备数据集
def create_dataset():

    #1获取训练集
    #参一：数据集路径，参二：是否是训练集，参三：数据预处理，张量数据，参四，是否联网下载
    train_dataset=CIFAR10(root='./data',train=True,transform=ToTensor(),download=True)
    #2获取测试集
    test_dataset=CIFAR10(root='./data',train=False,transform=ToTensor(),download=True)
    #3.返回数据集
    return train_dataset,test_dataset

#搭建神经网络
class ImageModel(nn.Module):
    def __init__(self):
        #1.1初始化父类成员
        super().__init__()
        #1.2搭建神经网络
        #第一个卷积层                                    卷积核大小
        self.conv1=nn.Conv2d(3,32,3,1,0)
        #第一个池化层 窗口大小2*2
        self.pool1=nn.MaxPool2d(2,2,0)
        #第二个卷积层
        self.conv2 = nn.Conv2d(32, 128, 3, 1, 0)
        self.pool2=nn.MaxPool2d(2,2,0)
        #第一个隐藏层（全连接层）576=16*6*6
        self.linear1=nn.Linear(128*6*6,512)
        #第二个隐藏层（全连接层）
        self.linear2 = nn.Linear(512, 512)
        #第三个隐藏层（全连接层）
        self.output = nn.Linear(512, 10)
        self.dropout=nn.Dropout(p=0.5)

    def forward(self,x):
        #第一层：卷积层+激活函数+池化层
        x=self.pool1(torch.relu(self.conv1(x)))
        # 第2层：卷积层+激活函数+池化层
        x = self.pool2(torch.relu(self.conv2(x)))
        # 第3层：全连接层+激活层  全连接层只能处理二维数据，所以要对数据进行拉平（8，16，6，6）-（8，576）
        #参一：样本数（行数），参二：列数，-1表示自动计算
        x=x.reshape(x.size(0),-1)
        x=torch.relu(self.linear1(x))
        x=self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        return self.output(x)



#模型训练

def train(train_dataset):
    #1创建数据加载器
    dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE)
    #2创建模型对象
    model=ImageModel()
    #3创建损失函数对象
    criterion=nn.CrossEntropyLoss()
    #4创建优化器对象
    optimizer=optim.Adam(model.parameters(),lr=0.0001)
    #5循环遍历epoch，开始每轮的训练工作
    epochs=10
    #遍历完成每轮的说有批次的训练动作
    for epoch_idx in range(epochs):
        #定义变量，记录总损失，总样本数据量，预测正确样本个数，训练时间
        total_loss,total_samples,total_correct,start=0.0,0,0,time.time()
        #遍历数据加载器，获取每批次的数据
        for x,y in dataloader:
            #切换训练模式
            model.train()
            #模型预测
            y_pred=model(x)
            #计算损失
            loss=criterion(y_pred,y)
            #梯度清零+反向传播+参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #预测正确的样本个数
            #print(y_pred)批次中每张图，每个分类的概率
            #print(torch.argmax(y_pred,dim=-1))#-1这里表示行
            #print(y)
            #print(torch.argmax(y_pred,dim=-1)==y)
            #print((torch.argmax(y_pred, dim=-1) == y).sum())
            total_correct+=(torch.argmax(y_pred,dim=-1)==y).sum()
            #统计当前批次的总损失 第一批的平均损失*第一批的样本个数
            total_loss+=loss.item()*len(y)
            #统计当前批次的总样本个数
            total_samples+=len(y)
        print(f'epoch:{epoch_idx+1}，loss：{total_loss/total_samples:5f},acc:{total_correct/total_samples:2f},time:{time.time()-start:.2f}')
    #保存模型
    torch.save(model.state_dict(),'./model/image_model.pth')



#模型测试
def evaluate(test_dataset):
    #创建测试集数据加载器
    dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    #创建模型对象
    model=ImageModel()
    #加载模型参数
    model.load_state_dict(torch.load('./model/image_model.pth'))
    total_correct,total_samples=0,0
    for x,y in dataloader:
        model.eval()
        y_pred=model(x)
        #因为训练的时候用了CROSSENTROPYLOSS，所以搭建神经网络时没有加softmax，这里要用argmax模拟
        #argmax函数功能：返回最大值对应的索引，充当该图片的预测分类
        y_pred=torch.argmax(y_pred,dim=-1)#-1表示行
        #5.4统计预测正确的样本个数
        total_correct+=(y_pred==y).sum()
        #5.5统计总样本个数
        total_samples+=len(y)
        #5.6打印正确率
    print(f'ACC:{total_correct/total_samples:.2f}')





#测试
if __name__ == '__main__':
    #1获取数据集
    train_dataset,test_dataset=create_dataset()
    #print(f'训练集：{train_dataset.data.shape}')   #(50000,32,32,3)
    #print(f'训练集：{test_dataset.data.shape}')    #(10000,32,32,3)
    #print(f'数据集的类别：{train_dataset.class_to_idx}')#数据类别用1-9的数据展示
    #2.搭建神经网络
    #model=ImageModel()
    #查看模型参数
    #summary(model,(3,32,32),batch_size=BATCH_SIZE)
    #模型训练
    #train(train_dataset)
    #模型评估
    evaluate(test_dataset)

