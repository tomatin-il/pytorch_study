'''序列数据：后面数据对前面数据有依赖，eg天气预测，股市分析，文本生成
组成：
词嵌入层、循环网络层、输出层、
词嵌入层：把词（或者词对应的索引）转换成词向量
'''

import torch
import jieba
import torch.nn as nn
#定义函数，用于演示 词嵌入层的API ，如何把词（词的索引）——>词向量
def dm01():
    #定义一句话
    text='北京东奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
    #分词
    words=jieba.lcut(text)
    print(f'分词结果：{words}')
    #创建词嵌入层
    #参一：词表大小（词的个数） 参二：词向量的维度
    embed=nn.Embedding(len(words),4)
    #获取每个词对象的下标索引
    i=0
    #for word in words:
    #    print(i,word)
    #   i+=1
    #enumerate():返回列表中每个值 及其对应的 索引
    i = 0
    for i, word in enumerate(words):
        #把词索引 转成 词向量
        word_vector=embed(torch.tensor(i)) #随机的每次都不一样
        print(f'词：{word},\t\t词向量:{word_vector} ')

#测试
if __name__ == '__main__':
    dm01()