#Cat 不改变维度数，拼接张量，除了拼接的那个维度外，其它维度数必须保持一致,eg:求t3时，列数必须都是3，但行数可以不一样
#stack()会改变维度数，拼接张量，所有的维度都必须保持一致
import torch
torch.manual_seed(4)

t1=torch.randint(1,10,size=(2,3))
print(f't1:{t1},shape:{t1.shape}')

t2=torch.randint(1,10,size=(2,3))
print(f't2:{t2},shape:{t2.shape}')

#t3=torch.cat([t1,t2],dim=0)
#print(f't3:{t3},shape:{t3.shape}')

#t4=torch.cat([t1,t2],dim=1)#维度数不能大于1
#print(f't4:{t4},shape:{t4.shape}')
print('-'*30)

t5=torch.stack([t1,t2],0)
print(f't5:{t5},shape:{t5.shape}')#按照0维度拼接，新加的2在最前面

t6=torch.stack([t1,t2],1)
print(f't6:{t6},shape:{t6.shape}')#按照1维度拼接，新加的2在中间

t7=torch.stack([t1,t2],2)#dim最大只能到2
print(f't7:{t7},shape:{t7.shape}')#按照2维度拼接，新加的2在最后


