import torch
import torch.nn as nn
from torchsummary import summary #计算模型参数，查看模型结构

#todo：1 搭建神经网络，即:自定义继承 nn.Module
class ModelDemo(nn.Module):
    #1.1在init魔法方法中，完成初始化：父类成员，及神经网络搭建，
    def __init__(self):
        #1.1初始化父类成员
        super().__init__()

        #1.2搭建神经网络隐藏层+输出层
        #隐藏层1：输入特征数3，输出特征数3
        self.linear1=nn.Linear(3,3)
        # 隐藏层2：输入特征数3，输出特征数2
        self.linear2 = nn.Linear(3, 2)
        # 输出层：输入特征数2，输出特征数2
        self.output = nn.Linear(2, 2)

        #1.3对隐藏层进行参数初始化
        #隐藏层1
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 隐藏层2
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    #1.2 前向传播：输入层-隐藏层-输出层
    def forward(self,x):
        #1.1 第一层 隐藏层计算：加权求和+激活函数
        #分解版写法
        #x=self.linear1(x)#加权求和
        #x=torch.sigmoid(x)#激活函数

        #合并版写法
        x=torch.sigmoid(self.linear1(x))
        # 1.2 第二层 隐藏层计算：加权求和+激活函数
        x = torch.relu(self.linear2(x))
        # 1.3 第三层 隐藏层计算：加权求和+激活函数
        x = torch.softmax(self.output(x),dim=-1)#dim=-1表示按行来计算
        #1.4返回预测值
        return x


#todo：2 模型训练
def train():
    #1创建模型对象
    my_model=ModelDemo()
    print(f'my_model:{my_model}')
    #创建数据集样本，随机生成
    data=torch.randn(size=(5,3))
    print(f'data:{data}')
    print(f'data.shape:{data.shape}')
    print(f'data.requires_grad:{data.requires_grad}')#是否自动微分
    #调用神经网络模型，进行模型训练
    output=my_model(data)#自动调用forward方法，进行前向传播
    print(f'output:{output}')
    print(f'output.shape:{output.shape}')
    print(f'output.requires_grad:{output.requires_grad}')
    print('-'*30)
    #4.打印计算和模型参数
    #参一：神经网络模型对象 参二：输入数据维度
    #计算模型参数
    summary(my_model,input_size=(5,3))
    #查看模型参数
    for name,param in my_model.named_parameters():
        print(f'name:{name}')
        print(f'param:{param}\n')
if __name__ == '__main__':
    train()






