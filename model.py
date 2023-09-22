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
    def __init__(self, head_size):
        super(SelfAttentionHead, self).__init__()    
        self.head_size = head_size
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        key = nn.Linear(x.size(dim=2), self.head_size, bias=False)
        query = nn.Linear(x.size(dim=2), self.head_size, bias=False)
        value = nn.Linear(x.size(dim=2), self.head_size, bias=False)
        
        k = key(x)
        q = query(x)
        w = q @ k.T(2, 1) * self.head_size**-0.5
        
        w = F.softmax(w, dim=1)
        w = self.dropout(w)
        
        v = value(x)
        out = w @ v
        
        return out
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, head_size, emb_size):
        self.heads = heads
        self.sa_heads = nn.ModuleList([SelfAttentionHead(head_size=head_size) for _ in range(self.heads)])
        self.projection = nn.Linear(head_size * heads, emb_size)
        self.dropout = nn.Dropout(0.2)
        self.ln = nn.LayerNorm(emb_size)
        
    def forward(self, x):
        out = torch.concat([h(self.ln(x)) for h in self.sa_heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
              
        return out
        
               
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
    def __init__(self, stride, ch, timestep_emb, word_emb, use_ca=False):
        super(DBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=stride),
        self.timestep_emb = timestep_emb.view(ch, -1, -1) # might not work...
        self.word_emb = word_emb.view(ch, -1, -1) # might not work...
        self.resnet = ResNetBlock(ch=ch),
        self.use_ca = use_ca
        self.mhsa = MultiHeadSelfAttention(heads=8, head_size=ch*2)
        
    def forward(self, x):
        out = self.conv(x)
        out = out + self.timestep_emb + self.word_emb
        out = self.resnet(out)
        if self.use_ca:
            out = self.mhsa(out)
            
        return out
        

class UNet(nn.Module):
    def __init__(self, caption_emb, emb_size, diff_steps):
        super(UNet, self).__init__()
        
        self.timestep_emb = nn.Embedding(num_embeddings=diff_steps, embedding_dim=emb_size)
        self.word_emb = caption_emb
        
        self.db1 = DBlock(timestep_emb=self.timestep_emb, word_emb=self.word_emb)
        self.db2 = DBlock(timestep_emb=self.timestep_emb, word_emb=self.word_emb)
        self.db3 = DBlock(timestep_emb=self.timestep_emb, word_emb=self.word_emb, use_ca=True)
        self.db4 = DBlock(timestep_emb=self.timestep_emb, word_emb=self.word_emb, use_ca=True)
        self.ub1 = UBlock(use_ca=True)
        self.ub2 = UBlock()
        self.ub3 = UBlock()
    
        
    def forward(self, x):
        db1  = self.db1(x)
        
        return out