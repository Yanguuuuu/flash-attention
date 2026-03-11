import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttention(nn.Module):
    def __init__(self, head_dim=8, tpsize=2):
        super().__init__()
        self.tpsize = tpsize
        self.head_dim = head_dim

    def forward_1d(self, scores, V_T):
        """
        scores: [seq_len_k] 一个query对所有key的分数
        V_T: [head_dim, seq_len_k]  V的转置
        返回: [head_dim] 输出向量
        """
        seq_len_k = len(scores)
        head_dim = V_T.shape[0]
        
        # 沿着head_dim维度切分
        block_size = head_dim // self.tpsize
        
        g_m = []
        g_d = []
        g_o = []
        
        # 1. 每个块处理一部分head_dim维度
        for tid in range(self.tpsize):
            start = tid * block_size
            end = start + block_size
            
            # 获取当前块对应的V的部分（某些head_dim维度）
            v_block = V_T[start:end, :]  # [block_size, seq_len_k]
            
            # 对当前块的所有head_dim维度，用相同的scores
            # 但每个head_dim维度有自己的累加器
            m = float('-inf')
            d = 0.0
            o = torch.zeros(block_size)  # 当前块的输出向量
            
            for k in range(seq_len_k):  # 遍历所有key
                s_k = scores[k].item()
                
                m_old = m
                m = max(m, s_k)
                
                if m_old > float('-inf'):
                    rescale = torch.exp(torch.tensor(m_old - m))
                else:
                    rescale = torch.tensor(1.0)
                
                exp_s = torch.exp(torch.tensor(s_k - m))
                
                # 更新分母
                d = d * rescale + exp_s
                
                # 更新分子 - 对当前块的所有head_dim维度同时更新
                o = o * rescale + exp_s * v_block[:, k]
            
            g_m.append(m)
            g_d.append(d.item())
            g_o.append(o)
        
        # 2. 合并所有块
        m_global = max(g_m)
        d_global = 0.0
        o_global = torch.zeros(head_dim)
        
        for i in range(self.tpsize):
            rescale = torch.exp(torch.tensor(g_m[i] - m_global))
            d_global += g_d[i] * rescale.item()
            
            # 将当前块的输出放到正确的位置
            start = i * block_size
            end = start + block_size
            o_global[start:end] += g_o[i] * rescale
        
        return o_global / d_global if d_global != 0 else torch.zeros(head_dim)
    
    def forward(self, S, V_T):
        """
        S: [seq_len_q, seq_len_k]
        V_T: [head_dim, seq_len_k]
        返回: [seq_len_q, head_dim]
        """
        seq_len_q, seq_len_k = S.shape
        head_dim, seq_len_k2 = V_T.shape
        assert seq_len_k == seq_len_k2
        
        O = torch.zeros(seq_len_q, head_dim)
        
        for i in range(seq_len_q):
            O[i] = self.forward_1d(S[i], V_T)
        
        return O

# 测试
if __name__ == "__main__":
    seq_len_q, seq_len_k, head_dim = 4, 4, 8
    
    S = torch.randn(seq_len_q, seq_len_k)
    V = torch.randn(seq_len_k, head_dim)
    V_T = V.T  # [head_dim, seq_len_k] 这就是你想要的
    
    print(f"S shape: {S.shape}")
    print(f"V_T shape: {V_T.shape}")
    
    fa = FlashAttention(head_dim=head_dim, tpsize=2)
    result = fa(S, V_T)
    
    print(f"结果 shape: {result.shape}")  # [4,8]
    
    # 验证
    traditional = torch.zeros(seq_len_q, head_dim)
    for i in range(seq_len_q):
        probs = F.softmax(S[i], dim=0)
        traditional[i] = torch.matmul(probs, V)  # V是[4,8]
    
    print(f"传统结果 shape: {traditional.shape}")
    print(f"最大误差: {torch.abs(result - traditional).max():.6f}")