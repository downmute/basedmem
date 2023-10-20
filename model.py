import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from custom_dataset import MemeDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import math
import numpy as np

class SelfAttentionHead(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(SelfAttentionHead, self).__init__()    
        self.hidden_size = hidden_size
        self.ln = nn.LayerNorm()
        self.context_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, hidden_size * 2)
        )
        self.key = nn.Linear(input_size, hidden_size, bias=False) 
        self.query = nn.Linear(input_size, hidden_size, bias=False) 
        self.value = nn.Linear(input_size, hidden_size, bias=False) 
        
    def forward(self, x, c_emb):
        c_emb = self.context_proj(c_emb)
        k = torch.concat([self.key(x), self.ln(c_emb)], dim=-1)
        q = self.query(x)
        w = q @ k.T(2, 1) * self.hidden_size**-0.5
        
        w = F.softmax(w, dim=1)
        
        v = torch.concat([self.value(x), self.ln(c_emb)], dim=-1)
        out = w @ v
        
        return out
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, hidden_size, emb_size, input_size):
        self.heads = heads
        self.sa_heads = nn.ModuleList([SelfAttentionHead(hidden_size=hidden_size, input_size=input_size) for _ in range(self.heads)])
        self.ln = nn.LayerNorm(emb_size)
        self.projection = nn.Linear(hidden_size * heads, emb_size)
        
        
    def forward(self, x, c_emb):
        out = torch.concat([h(self.ln(x), c_emb) for h in self.sa_heads], dim=-1)
        out = self.ln(self.projection(out))
              
        return out
        
               
class ResNetBlock(nn.Module):
    def __init__(self, ch):
        super(ResNetBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=ch, out_channels=ch,kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=ch, out_channels=ch,kernel_size=3),        
        ) 
        
        self.identity_conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1)
        
    def forward(self, x):
        identity = self.identity_conv(x)
        out = (identity + self.conv(x)) * 2**-0.5
        
        return out
    

class DBlock(nn.Module):
    def __init__(self, ch, t_emb_size, n_blocks, downsample=True, use_ca=False):
        super(DBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=2)
        self.ln = nn.LayerNorm()
        self.rn_blocks = nn.ModuleList([ResNetBlock(ch=ch) for _ in range(n_blocks)])
        self.use_ca = use_ca
        self.mhsa = MultiHeadSelfAttention(heads=8, hidden_size=64, emb_size=ch, input_size=ch)
        self.t_proj = nn.Linear(t_emb_size, ch)
        self.downsample = downsample
        
    def forward(self, t_emb, c_emb, x):
        t_proj = nn.SiLU(self.t_proj(t_emb))
        if self.downsample:
            out = self.conv(x)
        out = out + t_proj + self.ln(c_emb)
        for block in self.rn_blocks():
            out = block(out)
        if self.use_ca:
            out = self.ln(self.mhsa(out, c_emb))
            
        return out
    

class UBlock(nn.Module):
    def __init__(self, ch, t_emb_size, n_blocks, upsample=True, use_ca=False):
        super(UBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1),
        self.ln = nn.LayerNorm()
        self.rn_blocks = nn.ModuleList([ResNetBlock(ch=ch) for _ in range(n_blocks)])
        self.use_ca = use_ca
        self.mhsa = MultiHeadSelfAttention(heads=8, hidden_size=64, emb_size=ch, c_emb=self.c_emb)
        self.t_proj = nn.Linear(t_emb_size, ch)
        self.upsample = upsample
        
    def forward(self, t_emb, c_emb, x):
        t_proj = nn.SiLU(self.t_proj(t_emb))
        out = x + t_proj + self.ln(c_emb)
        for block in self.rn_blocks:
            out = block(out)
        if self.use_ca:
            out = self.ln(self.mhsa(out, c_emb))
        if self.upsample:
            out = self.conv(out)
            
        return out
        
class InceptionModule(nn.Module):
    def __init__(self, dim_in, dim_out, stride: int=2):
        super(InceptionModule, self).__init__()
        
        kernel_sizes = (3, 7, 15)
        num_scales = 3
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride)//2))
            
    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)        
        

class UNet(nn.Module):
    def __init__(self, diff_steps=400, t_emb_size=512):
        super(UNet, self).__init__()
        
        ## 64x64
        self.init_conv = InceptionModule()
        
        ## upsample
        self.t_emb  = self.get_timestep_embedding(diff_steps, t_emb_size)
        self.db1 = DBlock(ch=128, t_emb_size=t_emb_size, n_blocks=2)
        self.db2 = DBlock(ch=256, t_emb_size=t_emb_size, n_blocks=4)
        self.db3 = DBlock(ch=512, t_emb_size=t_emb_size, n_blocks=8)
        self.db4 = DBlock(ch=1024, t_emb_size=t_emb_size, n_blocks=8, downsample=False, use_ca=True)
        self.ub1 = UBlock(ch=1024, t_emb_size=t_emb_size, n_blocks=8, se_ca=True)
        self.ub2 = UBlock(ch=512, t_emb_size=t_emb_size, n_blocks=8)
        self.ub3 = UBlock(ch=256, t_emb_size=t_emb_size, n_blocks=4,)
        self.ub4 = UBlock(ch=128, t_emb_size=t_emb_size, n_blocks=2, upsample=False)
        self.conv_out = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3)
        
    def forward(self, c_emb, x_noisy, step):
        t_emb = self.t_emb(step)
        db1 = self.db1(t_emb, c_emb, x_noisy)
        db2 = self.db2(t_emb, c_emb, db1)
        db3 = self.db3(t_emb, c_emb, db2)
        db4 = self.db4(t_emb, c_emb, db3)
        ub1 = self.ub1(t_emb, c_emb, db4 + db4)
        ub2 = self.ub2(t_emb, c_emb, ub1 + db3)
        ub3 = self.ub3(t_emb, c_emb, ub2 + db2)
        ub4 = self.ub4(t_emb, c_emb, ub3 + db1)
        out = self.conv_out(ub4)
        
        return out
    
    def get_timestep_embedding(timesteps, embedding_dim: int):
        """
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        timesteps = torch.tensor(timesteps, dtype=torch.float32)
        assert len(timesteps.shape) == 1  

        half_dim = embedding_dim // 2
        emb = torch.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.pad(emb, (0, 0, 0, 1))
        assert emb.shape == [timesteps.shape[0], embedding_dim]
        return emb
    
    
class Imagen(nn.Module):
    def __init__(self, steps):
        super(Imagen, self).__init__()

        self.diff_steps = steps
        self.encoder = SentenceTransformer('sentence_transformers/sentence-t5-base')
        self.unet = UNet(diff_steps=self.diff_steps)
        
    def forward(self, prompt, x_noisy, step):
        enc_c_emb = self.encoder(prompt)    
        self.unet(enc_c_emb, x_noisy, step)
            

class ImagenTrainer():
    def __init__(self, csv_file, root_dir, epochs, img_size, beta1=0.0001, beta2=0.02, diff_steps=200):
        self.T = diff_steps
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.model = Imagen(diff_steps=self.T)
        self.noise_sch = torch.cos()
        self.img_size = img_size
        
        alphas, betas = self.cosine_noise_scheduler(diff_steps)
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
    def train(self):
        
        for _ in range(self.epochs):
        
            self.model()
    
    ## cosine scheduling is much more gradual than linear scheduling
    def cosine_noise_scheduler(timesteps):
        alphas = [
            torch.cos((t/timesteps+0.008)/1.008*(torch.tensor(math.pi)/2))**2 // \
            torch.cos(0.008/1.008*(torch.tensor(math.pi)/2))**2 for t in range(1, timesteps)]
        
        betas = [
            torch.clamp(1-((torch.cos((t/timesteps+0.008)/1.008*(torch.tensor(math.pi)/2))**2 // \
            torch.cos(0.008/1.008*(torch.tensor(math.pi)/2))**2) // \
            (torch.cos((t-1/timesteps+0.008)/1.008*(torch.tensor(math.pi)/2))**2 // \
            torch.cos(0.008/1.008*(torch.tensor(math.pi)/2))**2)), max=0.999) for t in range(1, timesteps)]
            
        return torch.tensor(alphas), torch.tensor(betas)
    
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
        
        dataset = MemeDataset(csv_file=self.csv_file, root_dir=self.root_dir, transform=data_transform)
        train_set, test_set = torch.utils.data.random_split(dataset, [1200, 270]) 
        
        return torch.utils.data.ConcatDataset([train_set, test_set])     
      
    def show_tensor_image(image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])
        
        ## only get the first image of batch
        if len(image.shape) == 4:
            image = image[0, :, :, :]
        plt.imshow(reverse_transforms(image))
        
    def get_loss(self, model, prompt, x_0, t):
        """
        We are using L2 loss, using summing reduction because mean reduction would be mse loss
        L2 loss should lead to higher sample diversity than L1 loss
        """
        x_noisy, noise = self.forward_diff(x_0, t)
        noise_pred = model(prompt, x_noisy, t)
        return F.mse_loss(noise, noise_pred, reduction='sum')
    
    def sample():
       
    

