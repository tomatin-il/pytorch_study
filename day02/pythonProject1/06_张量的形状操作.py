#reshape unsqueeze squeeze transpose permute view contiguous
import torch
torch.manual_seed(4)

#reshpae 在不改变张量内容的情况下，对其形状做改变
def dm01():
    t1=torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape},row:{t1.shape[0]},columns:{t1.shape[1],t1.shape[-1]}')#-1代表最后一维的大小
   # t2=t1.reshape(3,2)
   # t2 = t1.reshape(1, 6)
    t2 = t1.reshape(6, 1)
    print(f't2:{t2},shape:{t2.shape},row:{t2.shape[0]},columns:{t2.shape[1], t2.shape[-1]}')
    #t3=t1.reshape(2,5)不能改变数据个数

#squeeze删除所有为1的维度（降维）     unsqueeze 在指定的轴上增加一个1维度（升维）
def dm02():
    t1=torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape},row:{t1.shape[0]},columns:{t1.shape[1], t1.shape[-1]}')
    t2=t1.unsqueeze(0)#在0维上，增加一个维度，增加一个中括号
    print(f't2:{t2},shape:{t2.shape},row:{t2.shape[0]},columns:{t2.shape[1],t2.shape[-1]}')
    t3 = t1.unsqueeze(1)#在1维上，增加一个维度
    print(f't3:{t3},shape:{t3.shape},row:{t3.shape[0]},columns:{t3.shape[1], t1.shape[-1]}')
    t4 = t1.unsqueeze(2)#在2维上，增加一个维度，最大只能写到2
    print(f't4:{t4},shape:{t4.shape},row:{t4.shape[0]},columns:{t4.shape[1], t4.shape[-1]}')
    t5=torch.randint(1,10,size=(2,1,3,1,1))
    print(f't5:{t5},shape:{t5.shape},row:{t5.shape[0]},columns:{t5.shape[1], t5.shape[-1]}')
    t6=t5.squeeze()
    print(f't6:{t6},shape:{t6.shape},row:{t6.shape[0]},columns:{t6.shape[1], t6.shape[-1]}')

#transpose一次只能交换两个维度 permute一次同时可以交换多个维度
def dm03():
    t1=torch.randint(1,10,size=(2,3,4))
    print(f't1:{t1},shape:{t1.shape}')
    print('-'*30)
    t2=t1.transpose(0,1)
    print(f't1:{t1},shape:{t1.shape}')#不会改变原始数据
    print(f't2:{t2},shape:{t2.shape}')
    t3=t1.permute(2,0,1)
    print(f't3:{t3},shape:{t3.shape}')

#view只能修改连续的张量的形状，连续张量=内存中存储顺序和在张量中显示的顺序相同 修改完还是连续的
# contiguous把不连续的张量转变成连续的张量，即：基于张量中显示的顺序，修改内存中的存储顺序
# is_contiguous 判断张量是否是连续的
def dm04():
    t1=torch.randint(1,10,size=(2,3))
    print(t1.is_contiguous())
    t2=t1.view(3,2)#数据顺序没变
    print(f't1:{t1},shape:{t1.shape}')
    print(f't2:{t2},shape:{t2.shape}')
    print(t2.is_contiguous())
    t3=t1.transpose(0,1)
    print(f't3:{t3},shape:{t3.shape}')
    print(t3.is_contiguous())
    t4=t3.contiguous().view(2,3)

    print(f't4:{t4},shape:{t4.shape}')
if __name__ == '__main__':
    dm02()