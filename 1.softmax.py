# 用于实现safe_softmax激活函数
import torch.nn as nn
import torch
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
    
# 算子融合实现MLP层中Gate与Up融合 --- MLP层的计算
class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__(self)

    def forward(self,x:torch.Tensor): # x的shape为[B,seq_len, intermediate_dim * 2]
        x,y = x.chunk(2,-1) # 我参数运算可以合并
        return F.silu(x) * y # *为elementwise, @ 为矩阵乘法
    
if __name__ == "__main__":
    x = torch.rand([1,3,2,4],requires_grad=False)  # bacth,seq_len,num_head,hidden_dim
    softmax = Softmax()
    s1 = softmax(x)
    s2 = F.softmax(x,dim=-1)
    print(s1,'\n',s2)







