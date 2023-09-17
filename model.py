import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttentionHead(nn.Module):
    def __init__(self, head_size=16):
        super(MaskedAttentionHead, self).__init__()    
        self.heads = head_size
        
    def forward(self, x):
        key = nn.Linear(x.size(dim=2), self.heads, bias=False)
        query = nn.Linear(x.size(dim=2), self.heads, bias=False)
        value = nn.Linear(x.size(dim=2), self.heads, bias=False)
        
        k = key(x)
        q = query(x)
        w = q @ k.T(2, 1) * self.heads**-0.5  # dividing by sqrt of head size for variance near 1   
        
        tril = torch.tril(torch.ones(x.size(dim=1), x.size(dim=1)))
        w = w.masked_fill(tril == 0, float('-inf'))
        w = F.softmax(w, dim=1)
        
        v = value(x)
        out = w @ v
        
        return out
  
    
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size=16):
        super(SelfAttentionHead, self).__init__()    
        self.head_size = head_size
        
    def forward(self, x):
        key = nn.Linear(x.size(dim=2), self.head_size, bias=False)
        query = nn.Linear(x.size(dim=2), self.head_size, bias=False)
        value = nn.Linear(x.size(dim=2), self.head_size, bias=False)
        
        k = key(x)
        q = query(x)
        w = q @ k.T(2, 1) * self.head_size**-0.5
        
        w = F.softmax(w, dim=1)
        
        v = value(x)
        out = w @ v
        
        return out
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, head_size):
        self.heads = heads
        self.sa_heads = nn.ModuleList([SelfAttentionHead(head_size=head_size) for _ in range(self.heads)])
    
    def forward(self, x):
        heads_out = torch.concat([h(x) for h in self.sa_heads], dim=-1)
        out = heads_out + x
              
        return
        
               
class ResNetBlock(nn.Module):
    def __init__(self, ch):
        super(ResNetBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.GroupNorm(),
            nn.SiLU(),
            nn.Conv2d(in_channels=ch, out_channels=ch,kernel_size=3),
            nn.GroupNorm(),
            nn.SiLU(),
            nn.Conv2d(in_channels=ch, out_channels=ch,kernel_size=3),        
        ) 
        
        self.identity_conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1)
        
    def forward(self, x):
        identity = self.identity_conv(x)
        out = self.conv(x) + identity
        
        return out
    

class DBlock(nn.Module):
    def __init__(self, stride, ch):
        super(DBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=stride),
            # combine embs
            ResNetBlock(ch=ch),
            ResNetBlock(ch=ch),
            MultiHeadSelfAttention(heads=8, head_size=ch*2),
        )
        