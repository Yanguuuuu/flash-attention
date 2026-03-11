import torch
import torch.nn as nn
# 层归一化  # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ

# input:[bs,seq,num_heads,head_dim]
class RmsNorm(nn.Module):
    def __init__(self,gamma,eps):
        super().__init__()
        self.weight = nn.Parameter(gamma.detach().clone()) # rms_norm的放缩因子
        self.eps = eps

    def rms_forward(self,x):
        var = x.pow(2).mean(dim = -1,keepdim = True) + self.eps # 将最后维度进行广播
        sq_var = var.sqrt()
        return x / sq_var * self.weight
    
    def rms_residual(self,x,residual):
        x = x + residual
        return self.rms_forward(x)
    
    def forward(self,x,residual = None):
        if residual:
            return self.rms_residual(x,residual)
        else:
            return self.rms_forward(x)

class LayerNorm(nn.Module):
    def __init__(self,gamma,eps = 1e-5):
        super().__init__()
        self.weight =torch.nn.Parameter(gamma.detach().clone()) # 可学习点积张量
        self.eps = eps
    def gamma(self):
        return self.weight

    @torch.compile
    def rms_forward(self,x):
        var = x.pow(2).mean(dim = -1,keepdim = True) + self.eps
        sqrt_var = var.sqrt()
        x_norm = x / sqrt_var * self.weight
        return x_norm

    def residual_forward(self,x,residual):
        x = x + residual
        return self.rms_forward(x),x

    def forward(self,x,residual):
        if residual is not None:
            return self.residual_forward(x,residual)
        else:
            print(1)
            return self.rms_forward(x)

if __name__ == "__main__":
    x = torch.rand([1,3,2,4],requires_grad=False)  # bacth,seq_len,num_head,hidden_dim
    ln = LayerNorm(x,1e-5)
    rms = RmsNorm(x,1e-5)
    rms_r = ln(x,None)
    l1 = rms(x,None)
    print(rms_r,'\n',l1)