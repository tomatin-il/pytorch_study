#type(torch支持的数据类型)
#half[float16] double[float32] float[float64] short[int16] int[int32] long[int64]

import torch
#场景一：直接创建指定类型的张量
t1=torch.tensor([1,2,3,4,5],dtype=torch.float)
print(f't1:{t1},元素类型：{t1.dtype},张量类型：{type(t1)}')
print('-'*30)
#场景二：创建好张量后做类型转换
t2=t1.type(torch.int32)
print(f't2:{t2},元素类型：{t2.dtype},张量类型：{type(t2)}')
print('-'*30)
print(t2.half())