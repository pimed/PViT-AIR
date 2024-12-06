import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_



####################################################################################################################################
class PatchEmbedding(nn.Module):
    # NOTE: if number of channels of given input = k*in_chans > splits input image to k images and average over resulting patches 
    def __init__(self, image_size=224, embed_dim=256, patch_size=7, stride=4, in_chans=3, pos_embedding=None, compression=False):
        super().__init__()      
        self.image_size = image_size  
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.padding = patch_size//2

        self.compression = compression    
        self.pos_embedding = None if pos_embedding is None else PosEmbedding(pos_embedding)        # Position embedding 

        # Initialization for Patch embedding 
        self.conv = nn.Conv2d(in_chans, self.embed_dim, kernel_size=self.patch_size, stride=self.stride, padding=self.padding, bias=False)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.gelu = nn.GELU()          
    #-----------------------------------------------------------------------------------------------------------------------------
    def forward(self, *images):
        patches = list()
        for img in images:
            patch, H, W = self.forward_singleImage( img )             # Patch Embeding
            patches.append(patch)
        patch = torch.nansum( torch.stack(patches, dim=0), dim=0) 
        # patch = torch.nanmean( torch.stack(patches, dim=0), dim=0) 

        return patch, H, W
    #-----------------------------------------------------------------------------------------------------------------------------
    def forward_singleImage(self, image):
        '''
            Input:  image (B*C*H*W)
            Output: patch (B*N*C)
        '''        
        patch = self.conv(image)
        _, _, H, W = patch.shape
        patch = patch.flatten(2).transpose(1, 2) # BCHW -> BNC
        patch = self.norm(patch)
        patch = self.gelu(patch)

        # NOTE: position embedding before compression
        if self.pos_embedding is not None:
            patch = self.pos_embedding(patch)

        if self.compression:
            mask_kernel = torch.ones((1, self.in_chans, self.patch_size, self.patch_size)).to(image.device)
            mask = nn.functional.conv2d(image, mask_kernel, stride=self.stride, padding=self.padding) > .1     # BHW -> BN
            patch, H, W = self.compress(patch, H, W, mask)

        return patch, H, W
    #-----------------------------------------------------------------------------------------------------------------------------
    def compress(self, patch, H, W, mask):
        B,N,C = patch.shape
        patch_ = patch.view(B,H,W,C)
        mask_ = mask.view(B,H,W)

        patch_new, H_new, W_new = list(), 0, 0
        for b in range(B):
            y,x = torch.where(mask_[b])
            x1,x2,y1,y2 = min(x), max(x), min(y), max(y)
            h,w = y2-y1, x2-x1            
            p_new = patch_[b,y1:y2, x1:x2].reshape(h*w,-1)    
            H_new, W_new = max(H_new,h), max(W_new,w)
            patch_new.append(p_new)
            
        patch_new = torch.cat( [nn.functional.pad(s.permute(1,0), (0,H_new*W_new-s.size(0)), "constant", 0).permute(1,0).unsqueeze(0) for s in patch_new], dim=0)

        return patch_new, H_new, W_new    
####################################################################################################################################
class PosEmbedding(nn.Module):
    def __init__(self, emb_type='sin2D'):
        super().__init__()
        self.emb_type = emb_type
        self.initialized = False
    #-----------------------------------------------------------------------------------------------------------------------------      
    def forward(self, patch):  
        if not self.initialized: 
            self.initialize(patch)

        if self.pe is None: 
            return patch
        
        return patch + self.pe.repeat((patch.shape[0],1,1)).to(patch.device)
    #-----------------------------------------------------------------------------------------------------------------------------
    def initialize(self, patch):
        if self.emb_type == 'sin2D':
            B,N,C = patch.shape
            patch_size = int( N**0.5 )                                         # H, W are the same
            self.pe = self.sine2D(patch_size, C).flatten(1).transpose(0,1)     # BCHW -> BNC
       
        else:
            self.pe = None

        self.initialized = True         
    #-----------------------------------------------------------------------------------------------------------------------------      
    def sine2D(self, image_size, embed_dim):
        """
        Reference: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py 
        
        INPUTS: 
            image_size: height/width of the positions
            embed_dim: dimension of the model
        OUTPUT: 
            embed_dim*image_size*image_size position matrix
        """
        if embed_dim % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(embed_dim))
        pe = torch.zeros(embed_dim, image_size, image_size)
        # Each dimension use half of embed_dim
        embed_dim = int(embed_dim / 2)
        div_term = torch.exp(torch.arange(0., embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pos_w = torch.arange(0., image_size).unsqueeze(1)
        pos_h = torch.arange(0., image_size).unsqueeze(1)
        pe[0:embed_dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, image_size, 1)
        pe[1:embed_dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, image_size, 1)
        pe[embed_dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, image_size)
        pe[embed_dim + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, image_size)

        return pe 
####################################################################################################################################
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2, dropout=0.0, bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self._reset_parameters(bias)
    #-----------------------------------------------------------------------------------------------------------------------------            
    def forward(self, patch, patch_=None, return_attention=False):
        if patch_ is None: patch_ = patch
        B, N, C = patch.shape

        k = self.to_k(patch).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        v = self.to_v(patch).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        q = self.to_q(patch_).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3).reshape(B, N, self.embed_dim)
        output = self.to_out(values)
        output = self.drop(output)

        if return_attention:
            return output, attention
        else:
            return output        
    #-----------------------------------------------------------------------------------------------------------------------------            
    def _reset_parameters(self,  bias):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)

        if bias:
            self.to_q.bias.data.fill_(0)
            self.to_k.bias.data.fill_(0)
            self.to_v.bias.data.fill_(0)
            self.to_out.bias.data.fill_(0)
    #-----------------------------------------------------------------------------------------------------------------------------            
    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        scores = (q @ k.transpose(-2, -1)) 
        scores = scores / math.sqrt(d_k)
        attention = nn.functional.softmax(scores, dim=-1)
        values = torch.matmul(attention, v)

        return values, attention
####################################################################################################################################
class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, dropout=0.0, bias=False):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        
        self.gelu = nn.GELU()  
        self.drop = nn.Dropout(dropout)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(in_dim, hidden_dim) 
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=bias, groups=hidden_dim)
    #-----------------------------------------------------------------------------------------------------------------------------                    
    def forward(self, patch, H, W):
        patch = self.fc1(patch)
        patch = self.drop(patch)
        patch = self.gelu(patch)

        patch = self.conv( patch.transpose(1, 2).view(patch.size(0), patch.size(2), H, W)).flatten(2).transpose(1, 2)
        patch = self.drop(patch)
        patch = self.gelu(patch)
        
        patch = self.fc2(patch)
        patch = self.drop(patch)
        return patch
####################################################################################################################################
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0., bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.self_attn = MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, bias=bias)
        self.ff = FeedForward(dim, dropout=dropout, bias=bias)
    #-----------------------------------------------------------------------------------------------------------------------------            
    def forward(self, patch, H, W):
        # Self-Attention
        patch = patch + self.self_attn(patch) 
        patch = self.norm1(patch)
        # Feedforward
        patch = patch + self.ff(patch, H, W)
        patch = self.norm2(patch)

        return patch
    #-----------------------------------------------------------------------------------------------------------------------------            
    def forward_new(self, patch, H, W):
        # Self-Attention
        patch = self.norm1(patch)
        patch = patch + self.self_attn(patch) 
        # Feedforward
        patch = self.norm2(patch)        
        patch = patch + self.ff(patch, H, W)

        return patch
####################################################################################################################################
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0., bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.self_attn = MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, bias=bias)
        self.cross_attn = MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, bias=bias)
        self.ff = FeedForward(dim, dropout=dropout, bias=bias)
    #-----------------------------------------------------------------------------------------------------------------------------            
    def forward(self, patch, patch_, H, W):
        ''' patch_: encoder output '''
        # Self-Attention
        patch = patch + self.self_attn(patch) 
        patch = self.norm1(patch)
        # Cross-Attention
        patch = patch + self.cross_attn(patch, patch_) 
        patch = self.norm2(patch)        
        # Feedforward
        patch = patch + self.ff(patch, H, W)
        patch = self.norm3(patch)
        return patch
    #-----------------------------------------------------------------------------------------------------------------------------            
    def forward_new(self, patch, patch_, H, W):
        ''' patch_: encoder output '''
        # Self-Attention
        patch = self.norm1(patch)
        patch = patch + self.self_attn(patch) 
        # Cross-Attention
        patch = self.norm2(patch)        
        patch = patch + self.cross_attn(patch, patch_) 
        # Feedforward
        patch = self.norm3(patch)
        patch = patch + self.ff(patch, H, W)
        return patch    
####################################################################################################################################
class Transformer(nn.Module):
    def __init__(self, image_size=224, in_chans=3, patch_size=7, embed_dim=256, stride=4, num_layers=3, num_heads=2, dropout=0., bias=False, pos_embedding=False, device='cpu'):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        # p_diff = image_size % patch_size 
        # l = 0 if p_diff == 0 else patch_size - p_diff
        # self.padding = torch.nn.ZeroPad2d((l//2, l-l//2, l//2, l-l//2))

        self.device = device
        self.patch_emb = PatchEmbedding(embed_dim=embed_dim, patch_size=patch_size, stride=stride, in_chans=in_chans, pos_embedding=pos_embedding).to(device)   # Patch embedding
        self.encoders = nn.ModuleList([ EncoderBlock(embed_dim, num_heads=num_heads, dropout=dropout, bias=bias).to(self.device) for _ in range(num_layers)])   # Tranformer encoder    
    
        self.apply(_init_weights)
    #-----------------------------------------------------------------------------------------------------------------------------            
    def forward(self, *images):
        images_ = [image.to(self.device) for image in images]
        # images_ = [self.padding(image).to(self.device) for image in images]

        patch, H, W = self.patch_emb(*images_)                          # Patch Embeding
        for encoder in self.encoders: patch = encoder(patch, H, W)      # Transformer encoder
        return patch
####################################################################################################################################
class CrossTransformer(nn.Module):
    def __init__(self, image_size=224, in_chans=3, patch_size=7, embed_dim=256, stride=4, num_layers=3, num_heads=2, dropout=0., bias=False, pos_embedding=False, device='cpu'):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        # p_diff = image_size % patch_size 
        # l = 0 if p_diff == 0 else patch_size - p_diff
        # self.padding = torch.nn.ZeroPad2d((l//2, l-l//2, l//2, l-l//2))

        self.device = device
        self.patch_emb = PatchEmbedding(embed_dim=embed_dim, patch_size=patch_size, stride=stride, in_chans=in_chans, pos_embedding=pos_embedding).to(device)                                # Patch embedding

        self.encoders = nn.ModuleList([ EncoderBlock(embed_dim, num_heads=num_heads, dropout=dropout, bias=bias).to(self.device) for _ in range(num_layers)])   # Tranformer encoder    
        self.decoders = nn.ModuleList([ DecoderBlock(embed_dim, num_heads=num_heads, dropout=dropout, bias=bias).to(self.device) for _ in range(num_layers)])   # Tranformer decoder               
      
        self.apply(_init_weights)
    #-----------------------------------------------------------------------------------------------------------------------------            
    def forward(self, image, image_):
        patch, H, W = self.patch_emb(image.to(self.device))
        patch_, _, _ = self.patch_emb(image_.to(self.device))

        # patch, H, W = self.patch_emb( self.padding(image).to(self.device))
        # patch_, _, _ = self.patch_emb( self.padding(image_).to(self.device))

        # Transformer encoder/decoder
        for (encoder, decoder) in zip(self.encoders, self.decoders):
            patch = encoder(patch, H, W)
            patch_ = decoder(patch_, patch, H, W)

        return patch_ 
    #-----------------------------------------------------------------------------------------------------------------------------            
    def forward_(self, image, image_):
        # Patch Embeding
        patch, H, W = self.patch_emb(image.to(self.device))
        patch_, _, _ = self.patch_emb(image_.to(self.device))

        # Transformer encoder/decoder
        for encoder in self.encoders:
            patch = encoder(patch, H, W)

        for decoder in self.decoders:
            patch_ = decoder(patch_, patch, H, W)

        return patch_     
####################################################################################################################################
def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_() 
####################################################################################################################################
