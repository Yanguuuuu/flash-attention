# flashattention的模拟实现--支持行分块计算  tid表示:threadIdx
# onlinesoftmax的模拟实现
# rmsnorm 层归一化是爱你

'''  
for i,val in enumerate(s):       # online softmax同时更新 分子分母 res = o / d
        m_old = m
        m = max(m, val)
        d = d * torch.exp(m_old - m) + torch.exp(val - m)
        o = o * torch.exp(m_old - m) + torch.exp(val - m) * v[i]
'''
