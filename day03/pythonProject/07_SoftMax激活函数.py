import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#定义张量，记录：分类数据（把值映射成概率，最后概率值相加等于1）
scores=torch.tensor([0.2,0.02,0.15,0.15,1.3,0.5,0.06,1.1,0.05,3.75])
probabilities=torch.softmax(scores,dim=0)#按行计算
print(probabilities)