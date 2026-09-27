#L1正则化：权重可以变为0，相当于：降维 L2正则化：权重可以无限接近0 DROPOut：随机失活，每批次样本训练时，随机让一部分神经元死亡，防止一些特征对结果的影响较大（防止过拟合）

import torch
import torch.nn as nn
#定义函数，演示：随即失活
def dm01():
    #创建隐藏层输出结果
    t1=torch.randint(0,10,size=(1,4)).float()
    print(f't1:{t1}')
    #创建先行层
    linear1=nn.Linear(4,5)
    #进行下一层加权求和和激活函数计算
    #加权求和
    l1=linear1(t1)
    print(f'l1:{l1}')
    #激活函数
    output=torch.relu(l1)
    print(f'output:{output}')
    #对激活值进行随即失活处理，只有训练阶段有，测试阶段没有
    dropout=nn.Dropout(p=0.5)#每个神经元都有50%的概率被kill
    d1=dropout(output)
    print(f'd1(随机失活后的数据)：{d1}')
if __name__ == '__main__':
    dm01()

