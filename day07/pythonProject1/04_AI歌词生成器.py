'''
用给定的起始词，结合长度，来进行ai歌词生成
1获取数据，进行分词，获取词表
2数据预处理，构建数据集
3.搭建RNN神经网络
4训练模型
5模型预测
'''
import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

#todo 获取数据，进行分词，获取词表
def build_vocab():
    #1定义变量，记录去重后的所有词,每行文本的分词结果
    unique_words,all_words=[],[]
    #2遍历数据集，获取到每行文本
    #r读数据，utf-8，码表
    for line in open('./data/jaychou_lyrics.txt','r',encoding='utf-8'):
        #获取每行歌词，进行分词
        words=jieba.lcut(line)
        #所有分词结果记录到all_words中
        all_words.append(words)
        #2.3遍历分词结果，去重后添加到unique_words中
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    #3统计去重后词的数量
    word_count=len(unique_words)
    #print(word_count)#5631
    #4构建词表，字典形式，key是词，value是词的索引
    word_to_index={word:i for i,word in enumerate(unique_words)}
    #print(f'word_to_index:{word_to_index}')
    #5歌词文本用词表索引表示
    corpus_idx=[]
    #6遍历每一行的分词结果
    for words in all_words:
        #6.1定义变量,记录词索引列表
        tmp=[]
        #获取每一行的词对应的索引
        for word in words:
            tmp.append(word_to_index[word])
        #6.3在每行词中间添加空格
        tmp.append(word_to_index[' '])
        #6.4获取文档中每个词的索引，添加到corpus_idx
        corpus_idx.extend(tmp)
        #print(f'corpus_idx:{corpus_idx}')
    #7返回结果：唯一词列表，词表,去重后词的数量，歌词用索引表示
    return unique_words,word_to_index,word_count,corpus_idx

#todo 数据预处理。构建数据集
class LyricsDataset(torch.utils.data.Dataset):
    #初始化词索引，词个数等
    def __init__(self,corpus_idx,num_chars):
        #1.1文档数据中词的索引
        self.corpus_idx=corpus_idx
        #1.2每个句子中词的个数
        self.num_chars=num_chars
        #1.3文档中词的个数，不去重
        self.word_count=len(self.corpus_idx)
        #1.4句子数量
        self.number=self.word_count//num_chars

    #当使用len(obj)时，自动调用此方法
    def __len__(self):
        return self.number

    #当使用dataset（index）时，自动调用此方法
    def __getitem__(self, idx):
        #idx指的是词的索引，并将其修正索引值到文档的范围里面
        #3.1确保索引start在合法范围内，start：当前样本的起始索引
        start=min(max(idx,0),self.word_count-self.num_chars-1)
        #计算结束索引
        end=start+self.num_chars
        #输入值，从文档中取出start~到end的索引词作为x
        x=self.corpus_idx[start:end] #包左不包右
        #输出值，网络预测结果
        y=self.corpus_idx[start+1:end+1]
        #3.5返回输入值和输出值的张量形式
        return torch.tensor(x),torch.tensor(y)


#todo 搭建RNN网络
class TextGenerator(nn.Module):
    #1初始化方法
    def __init__(self,unique_word_count):
        #1.1初始化父类
        super().__init__()
        #1.2初始化词嵌入层：语料中词的数量，词向量的而维度
        self.ebd=nn.Embedding(unique_word_count,128)
        #1.3循环网络层：词向量维度，隐藏层维度：256，网络层数：1
        self.rnn=nn.RNN(128,256,1)
        #1.4输出层（全连接层）：特征向量维度（和隐藏层向量维度一致），词表中词的个数
        self.out=nn.Linear(256,unique_word_count)#词表中每个词的概率，选取概率最大的为预测结果

    #前向传播
    def forward(self,inputs,hidden):
        #2.1初始化，词嵌入层处理
        #embd格式：（batch句子的数量，句子的长度，词向量维度）
        embd =self.ebd(inputs)
        #rnn（句子的长度，batch句子的数量，隐藏层维度）
        output,hidden=self.rnn(embd.transpose(0,1),hidden)
        #2.3全连接，输入内容必须是二维数据，即：词的数量*词的维度128
        '''
        RNN 输出 `output` 的原始形状：`[seq_len, batch, hidden_dim]`
        `output.shape[-1]`：取最后一维，也就是隐藏层维度`hidden_dim`
        `.reshape(-1, output.shape[-1])`
        `-1`：PyTorch 自动算第一维 = `seq_len × batch`
       形变后：`[seq_len * batch , hidden_dim]`
        '''
        output=self.out(output.reshape(shape=(-1,output.shape[-1])))
        #2.4返回结果
        return output,hidden

    #隐藏层的初始化方法
    def init_hidden(self,bs):
        #隐藏层初始化：[网络层数，batch_size，隐藏层的向量维度]
        return torch.zeros(1,bs,256)



#todo 训练模型
def train():
    #1构建词典
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    #2获取数据集
    lyrics=LyricsDataset(corpus_idx,32)
    #3初始化神经网络
    model=TextGenerator(unique_word_count)
    #4创建数据加载器对象5(每批5个句子，每个句子32个词)
    lyrics_dataloader=DataLoader(lyrics,batch_size=5,shuffle=True)
    #5损失函数
    criterion=nn.CrossEntropyLoss()
    #6定义优化器
    optimizer=torch.optim.Adam(model.parameters(),0.001)
    #7模型训练
    epochs=10
    for epoch in range(epochs):
        #定义变量记录:本轮开始训练时间，迭代批次，训练总损失
        start,iter_num,total_loss=time.time(),0,0
        for x,y in lyrics_dataloader:
            bs=x.size(0)
            hidden=model.init_hidden(bs)
            output,hidden=model(x,hidden)
            #y的形状（batch,句子长度，词向量维度）
            #ouput形状（句子长度，batch，词向量维度）
            #先转换顺序，再转成一维向量，每个词的下标索引
            #y是要操作的对象，0和1是要交换的维度   -1表示自动计算总元素的数量y 从[batch,seq_len]变成一维张量 [seq_len × batch]
            y=torch.transpose(y,0,1).reshape(shape=(-1,))
            loss=criterion(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            iter_num+=1
        print(f'epoch:{epoch+1}，time:{time.time()-start:.2f},loss:{total_loss/iter_num:.4f}')

    torch.save(model.state_dict(),'./model/text_generator.pth')



#模型预测
def evaluate(start_word,sentence_length):
    #构建词典
    unique_words,word_to_index,unique_word_count,corpus_idx=build_vocab()
    #获取模型
    model=TextGenerator(unique_word_count)
    model.load_state_dict(torch.load('./model/text_generator.pth'))
    #获取隐藏层初始值
    hidden=model.init_hidden(1)
    #将输入的 开始词转换成索引
    word_idx=word_to_index[start_word]
    #定义列表，存放：产生的词的索引
    generate_sentence=[word_idx]
    for i in range(sentence_length):
        #7.1模型预测
        '''
         `[word_idx]` → `[5]` 一维列表
         `[[word_idx]]` → `[[5]]` 二维列表，构建形状 `(seq_len, batch_size)` 的 tensor
         第 1 维 `seq_len = 1`：一次输入1个单词（时间步 = 1）
         第 2 维 `batch_size = 1`：批次大小为 1，只 1 条样本    
        '''
        output,hidden=model(torch.tensor([[word_idx]]),hidden)
        #7.2获取预测结果
        word_idx=torch.argmax(output)
        #7.3把预测结果添加到列表中
        generate_sentence.append(word_idx)
    #将索引转成词
    for idx in generate_sentence:
        print(unique_words[idx],end='')




#测试
if __name__ == '__main__':
    #unique_words, word_to_index, word_count, corpus_idx=build_vocab()
    #print(f'词的数量：{word_count}')
    #print(f'去重后的词：{unique_words}')
    #print(f'每个词的索引：{word_to_index}')
    #print(f'文档中每个词对应的索引：{corpus_idx}')

    #构建数据集
    #dataset=LyricsDataset(corpus_idx,5)
    #print(f'句子数量：{len(dataset)}')
    #查看下 输入值和目标值
    #x,y=dataset[1]
    #print(f'输入值：{x}')
    #print(f'输入值：{y}')

    #train()

    evaluate('分手',50)