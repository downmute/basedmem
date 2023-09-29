import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from custom_dataset import MemeDataset
from torchvision import transforms

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, c_emb):
        super(SelfAttentionHead, self).__init__()    
        self.hidden_size = head_size
        self.dropout = nn.Dropout(0.2)
        self.c_emb = c_emb
        self.ln = nn.LayerNorm()
        
    def forward(self, x):
        key = nn.Linear(x.size(dim=2), self.hidden_size, bias=False) 
        query = nn.Linear(x.size(dim=2), self.hidden_size, bias=False) 
        value = nn.Linear(x.size(dim=2), self.hidden_size, bias=False) 
        
        k = torch.concat([key(x), self.ln(self.c_emb)], dim=-1)
        q = query(x)
        w = q @ k.T(2, 1) * self.hidden_size**-0.5
        
        w = F.softmax(w, dim=1)
        w = self.dropout(w)
        
        v = torch.concat([value(x), self.ln(self.c_emb)], dim=-1)
        out = w @ v
        
        return out
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, head_size, emb_size, c_emb):
        self.heads = heads
        self.sa_heads = nn.ModuleList([SelfAttentionHead(head_size=head_size, c_emb=c_emb) for _ in range(self.heads)])
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
        out = (identity + self.conv(x))* 2**-0.5
        
        return out
    

class DBlock(nn.Module):
    def __init__(self, stride, ch, t_emb_size, downsample=True, use_ca=False):
        super(DBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=stride),
        self.ln = nn.LayerNorm()
        self.rn1 = ResNetBlock(ch=ch),
        self.rn2 = ResNetBlock(ch=ch)
        self.use_ca = use_ca
        self.mhsa = MultiHeadSelfAttention(heads=8, head_size=ch*2, emb_size=ch, c_emb=self.c_emb)
        self.t_proj = nn.Linear(t_emb_size, ch)
        self.downsample = downsample
        
    def forward(self, t_emb, c_emb, x):
        t_proj = self.t_proj(t_emb)
        if self.downsample:
            out = self.conv(x)
        out = out + t_proj + self.ln(c_emb)
        out = self.rn1(out)
        out = self.rn2(out)
        if self.use_ca:
            out = self.ln(self.mhsa(out))
            
        return out
    

class UBlock(nn.Module):
    def __init__(self, stride, ch, t_emb_size, upsample=True, use_ca=False):
        super(UBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=stride),
        self.ln = nn.LayerNorm()
        self.rn1 = ResNetBlock(ch=ch),
        self.rn2 = ResNetBlock(ch=ch)
        self.use_ca = use_ca
        self.mhsa = MultiHeadSelfAttention(heads=8, head_size=ch*2, emb_size=ch, c_emb=self.c_emb)
        self.t_proj = nn.Linear(t_emb_size, ch)
        self.upsample = upsample
        
    def forward(self, t_emb, c_emb, x):
        t_proj = self.t_proj(t_emb)
        out = x + t_proj + self.ln(c_emb)
        out = self.rn1(out)
        out = self.rn2(out)
        if self.use_ca:
            out = self.ln(self.mhsa(out))
        if self.upsample:
            out = self.conv(out)
            
        return out
        

class UNet(nn.Module):
    def __init__(self, diff_steps=200, t_emb_size=16):
        super(UNet, self).__init__()
        
        self.t_emb = nn.Embedding(num_embeddings=diff_steps, embedding_dim=t_emb_size)
        self.db1 = DBlock(stride=1, ch=128, t_emb_size=t_emb_size, diff_steps=diff_steps)
        self.db2 = DBlock(stride=1, ch=256, t_emb_size=t_emb_size, diff_steps=diff_steps)
        self.db3 = DBlock(stride=1, ch=384, t_emb_size=t_emb_size, diff_steps=diff_steps, use_ca=True)
        self.db4 = DBlock(stride=1, ch=512, t_emb_size=t_emb_size, diff_steps=diff_steps, downsample=False, use_ca=True)
        self.ub1 = UBlock(stride=2, ch=512, t_emb_size=t_emb_size, diff_steps=diff_steps, use_ca=True)
        self.ub2 = UBlock(stride=2, ch=384, t_emb_size=t_emb_size, diff_steps=diff_steps, use_ca=True)
        self.ub3 = UBlock(stride=2, ch=256, t_emb_size=t_emb_size, diff_steps=diff_steps)
        self.ub4 = UBlock(stride=2, ch=128, t_emb_size=t_emb_size, diff_steps=diff_steps, upsample=False)
        self.conv_out = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3)
              
        
    def forward(self, c_emb, step, x):
        t_emb = self.t_emb(step)
        db1 = self.db1(t_emb, c_emb, x)
        db2 = self.db2(t_emb, c_emb, db1)
        db3 = self.db3(t_emb, c_emb, db2)
        db4 = self.db4(t_emb, c_emb, db3)
        ub1 = self.ub1(t_emb, c_emb, db4 + db4)
        ub2 = self.ub2(t_emb, c_emb, ub1 + db3)
        ub3 = self.ub3(t_emb, c_emb, ub2 + db2)
        ub4 = self.ub4(t_emb, c_emb, ub3 + db1)
        out = self.conv_out(ub4)
        
        return out
    
    
class Imagen(nn.Module):
    def __init__(self):
        super(Imagen, self).__init__()
    
        self.encoder = SentenceTransformer('sentence_transformers/sentence-t5-base')
        self.unet = UNet(diff_steps=self.diff_steps)
        
    def forward(self, step, x):
        enc_c_emb = self.encoder(x)
        
        
            

class ImagenTrainer():
    def __init__(self, dataset, epochs, img_size, beta1=0.0001, beta2=0.02, diff_steps=200):
        self.T = diff_steps
        self.dataset = dataset
        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.model = Imagen(diff_steps=self.T)
        self.betas = torch.linspace(self.beta1, self.beta2, self.T)
        self.img_size = img_size
        
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
    def train(self):
        
        for _ in range(self.epochs):
        
            self.model()
            
    def get_idx_from_list(vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device())
    
    def forward_diff(self, x_0, t, device='gpu'):
        ## returns the noise ver of img at specific timestep
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_idx_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_idx_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        ## return mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
        
    def load_transformed_dataset(self):
        data_transforms = [
            transforms.resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t*2) -  1) ## scale the tensors between [-1, 1]
        ]
        
        data_transform = transforms.Compose(data_transforms)
        train = torchvision.dataset.StanfordCars
        
    

