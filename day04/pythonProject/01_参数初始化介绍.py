#参数初始化的目的：1防止梯度消失或梯度爆炸 2.提高收敛速度 3.打破对称性
#无法打破对称性的：全0，全1，固定值
#可以打破对称性的：随机初始化，正态分布初始化，kaiming初始化，xavier初始化

import torch.nn as nn
#1均匀分布随机初始化
def dm01():
    #1.创建1个线性层，输入维度为5，输出维度3
    Linear=nn.Linear(5,3)
    #2.对权重（w）进行随机初始化，从0-1均匀分布产生参数
    nn.init.uniform_(Linear.weight)
    #3.对偏置（b）进行随机初始化，从0-1均匀分布产生参数
    nn.init.uniform_(Linear.bias)
    #打印生成结果
    print(Linear.weight.data)
    print(Linear.bias.data)

#2.固定初始化
def dm02():
    # 1.创建1个线性层，输入维度为5，输出维度3
    Linear = nn.Linear(5, 3)
    # 2.对权重（w）进行初始化，设置固定值为4
    nn.init.constant_(Linear.weight,4)
    # 3.对偏置（b）进行初始化，设置固定值为4
    nn.init.constant_(Linear.bias,4)
    # 打印生成结果
    print(Linear.weight.data)
    print(Linear.bias.data)

#全0初始化
def dm03():
    # 1.创建1个线性层，输入维度为5，输出维度3
    Linear = nn.Linear(5, 3)
    # 2.对权重（w）进行初始化，全0初始化
    nn.init.zeros_(Linear.weight)
    # 3.对偏置（b）进行初始化，全0初始化
    nn.init.zeros_(Linear.bias)
    # 打印生成结果
    print(Linear.weight.data)
    print(Linear.bias.data)

#全1初始化
def dm04():
    # 1.创建1个线性层，输入维度为5，输出维度3
    Linear = nn.Linear(5, 3)
    # 2.对权重（w）进行初始化，全1初始化
    nn.init.ones_(Linear.weight)
    # 3.对偏置（b）进行初始化，全1初始化
    nn.init.ones_(Linear.bias)
    # 打印生成结果
    print(Linear.weight.data)
    print(Linear.bias.data)

#正态分布初始化
def dm05():
    # 1.创建1个线性层，输入维度为5，输出维度3
    Linear = nn.Linear(5, 3)
    # 2.对权重（w）进行初始化,正态分布初始化（均值为0，标准差为1）
    nn.init.normal_(Linear.weight)
    # 3.对偏置（b）进行初始化
    nn.init.normal_(Linear.bias)
    # 打印生成结果
    print(Linear.weight.data)
    print(Linear.bias.data)

#kaiming初始化(不能初始化偏置)
def dm06():
    Linear = nn.Linear(5, 3)
   # nn.init.kaiming_normal_(Linear.weight)  #正态分布初始化
    nn.init.kaiming_uniform_(Linear.weight)  #均匀分布初始化
    print(Linear.weight.data)

#Xavier初始化(不能初始化偏置)
def dm07():
    Linear = nn.Linear(5, 3)
    #nn.init.xavier_normal_(Linear.weight)  #正态分布初始化
    nn.init.xavier_uniform_(Linear.weight)  #均匀分布初始化
    print(Linear.weight.data)

if __name__ == '__main__':
    dm07()