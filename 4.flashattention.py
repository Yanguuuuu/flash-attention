import torch
import torch.nn as nn
import torch.nn.functional as F

# 实现offline-softmax
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x): # 在hidden维度进行softmax e^(xi-xmax) / sum (e^(xj - xmax)) safe_softmax
        max_h,_ = torch.max(x,dim = -1,keepdim=True)# 在hidden维度求出最大值
        sum_h = torch.sum(torch.exp(x - max_h),dim = -1,keepdim=True)  # 对最后一维进行求指数和,max_h 广播
        
        o_proj = torch.exp(x - max_h) / sum_h 
        return o_proj
    
'''  
for val in x:       # online softmax同时更新
        m_old = m
        m = max(m, val)
        d = d * np.exp(m_old - m) + np.exp(val - m)
'''

class OnlineSoftmax(nn.Module):
    def __init__(self, output_size = 100,tpsize = 1,tid = 0):         #  沿着head_dim维度进行tp切分
        super().__init__()
        self.tpsize = tpsize            # 切分的总的卡数量
        self.tid = tid             # tp的张量0
        self.output_size = output_size  # 输出的维度形状
        self.max = [float('-inf') for _ in range(tpsize)] # 每个block块内单独技术
             
    # seq_len * seq_len [n,n]:[1,n] 实现一个block块内的online_softmax
    def forward_1d(self,input:torch.Tensor):
        origin_x = input.clone()
        origin_y = input.clone()

        '1.计算块内的最大值分母,添加到列表中'
        g_m = []
        g_d = []    # 计算每一个块的
        for tid in range(self.tpsize):      # 对于每个thread无需计算
            share_size =int(self.output_size // self.tpsize ) 
            start_id = int(tid * share_size ) 
            x = origin_x.narrow(-1,start_id,share_size)                # 按照最后一个维度进行数据切分 # 得到每一个block块
            print(x)

            m = float('-inf')
            d = 0                # sum
            for i in range(share_size):
                m_old = m   
                if x[...,i] > m:
                    m = x[...,i]
                d = d * torch.exp(m_old - m)+ torch.exp(x[...,i] - m)  # 单个block内的总和

            g_m.append(m)
            g_d.append(d)


        print('g_m:',g_m,'g_d:',g_d)
        '2.计算block块间的最大值'
        m = max(g_m)
        print(m)
        d = 0
        for i in range(len(g_m)):
            d += g_d[i] * torch.exp(g_m[i] - m) 


        '3.计算 attention输出'
        for tid in range(self.tpsize):
            share_size = int(self.output_size / self.tpsize )   
            start_id = int(tid * share_size )              
            x = origin_y.narrow(-1,int(start_id),int(share_size))                # 按照最后一个维度进行数据切分 # 得到每一个block块
            for i in range(share_size):
                #print(x[...,i],m,d)
                x[...,i] = torch.exp(x[...,i] - m) / d
                # 这里可以同步更新
                
        return origin_y
    
    def forward(self,input:torch.Tensor):
        shape = input.shape
        input = input.view(-1,self.output_size).clone()
        for i in range(input.shape[0]):
            input[i] = self.forward_1d(input[i])

        return input.reshape(shape)

class FlashAttention(nn.Module):
    def __init__(self, output_size = 2,tpsize = 1):         #  沿着head_dim维度进行tp切分
        super().__init__()
        self.tpsize = tpsize            # 切分的总的卡数量
        self.output_size = output_size  # 输出的维度形状

    # seq_len * seq_len [n,n]:[1,n] 实现一个block块内的online_softmax
    def forward_1d(self,S:torch.Tensor,V:torch.Tensor):              # S = Q @ K^T / d
        origin_x = S.clone()
        origin_y = S.clone()

        origin_v1 = V.clone()
        origin_v2 = V.clone()

        '1.计算块内的最大值分母,添加到列表中'
        g_m = []
        g_d = []    # 计算每一个块的
        g_o = []    # 计算出每一个块的输出
        for tid in range(self.tpsize):      # 对于每个thread无需计算，模拟不同GPU数据读入
            share_size =int(self.output_size // self.tpsize ) 
            start_id = int(tid * share_size ) 
            x = origin_x.narrow(-1,start_id,share_size)                # 按照最后一个维度进行数据切分 # 得到每一个block块
            v = origin_v1.narrow(-1,start_id,share_size)
            print(x)

            m = float('-inf')
            d = 0                # sum,分母
            o = 0                # 
            for i in range(share_size):
                m_old = m   
                if x[...,i] > m:
                    m = x[...,i]
                d = d * torch.exp(m_old - m)+ torch.exp(x[...,i] - m)  # 单个block内的总和
                o = o * torch.exp(m_old - m) + torch.exp(x[...,i] - m) * v[...,i]

            g_m.append(m)
            g_d.append(d)
            g_o.append(o)

        print('g_m:',g_m,'g_d:',g_d,'g_o:',g_o)

        '2.计算block块间的最大值'
        m = max(g_m)
        print(m)
        d = 0
        o = 0
        for i in range(len(g_m)):
            d += g_d[i] * torch.exp(g_m[i] - m) 
            o += g_o[i] * torch.exp(g_m[i] - m)

        return o / d

        # '3.计算 attention输出'
        # for tid in range(self.tpsize):
        #     share_size = int(self.output_size / self.tpsize )   
        #     start_id = int(tid * share_size )              
        #     x = origin_y.narrow(-1,int(start_id),int(share_size))                # 按照最后一个维度进行数据切分 # 得到每一个block块
        #     for i in range(share_size):
        #         #print(x[...,i],m,d)
        #         x[...,i] = torch.exp(x[...,i] - m) / d
        #         # 这里可以同步更新
                
        # return origin_y
    
    def forward(self,S:torch.Tensor,V:torch.Tensor):
        shape = S.shape
        S = S.view(-1,self.output_size).clone()
        shapeV = V.shape
        V = V.view(-1,self.output_size).clone()
        O = torch.zeros(S.shape[0],V.shape[0])

        for i in range(S.shape[0]):
            for j in range(V.shape[0]):
                O[i][j] = self.forward_1d(S[i],V[j])

        return O
    
    
# [batches,seq_len,num_head,head_dim]
if  __name__ == "__main__":
        # # online softmax验证
        # att_score = torch.rand([2,8]) #[seq_length,seq_length]
        # f1 = OnlineSoftmax(int(att_score.shape[-1]),tpsize=2)
        # print(f1(att_score))
        # print(F.softmax(att_score,-1))
        # f2 = Softmax()
        # print(f2(att_score))

        # flash_atten验证
        S = torch.rand([4,4])
        V = torch.rand([4,8])
      

        f3 = FlashAttention(output_size = S.shape[-1],tpsize=1)
        o = F.softmax(S,dim=-1) @ V
        print('\n',f3(S,V.T),'\n',o)

        

