#add sub mul div neg取反
#add_()加了下划线可以修改源数据

import torch
t1=torch.tensor([1,2,3])
#t2=t1.add(10)  #数值会和张量中的每一个值进行数值运算
#t2=t1.add_(10) #会修改源数据
#t2=t1+10 #不会修改t1

#t2=t1.div(2)

t2=t1.neg()
print(f't1:{t1}')
print(f't2:{t2}')
