'''
    VAE implementation based on Transformer
      - VAE adapted from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
      - Transformer arch adapted from CaiT, with layer scale and talking heads: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cait.py
'''

from random import randrange
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# classes

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        '''
            dim: model's dim 
            fn: an arbitrary nn
            depth: the No of layer; an Attention + a FeedForward is one layer
        '''
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
        
    def forward(self, x, **kwargs): 
        return self.fn(x, **kwargs) * self.scale  # element multiplication


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_ratio, dropout = 0.):
        '''
            dim: model's dim
            ffn_ratio: like 4 etc.
        '''
        super().__init__()
        hidden_dim = int(dim * ffn_ratio)  
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # by default work on the dim=-1
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # like relu
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        '''
            x: dim (batch, seq, dim)
            context: dim (batch, seq, dim)
        '''
        b, n, _, h = *x.shape, self.heads

        # (1) do layer norm; norm on dim dimension
        x = self.norm(x) 
        context = x if context is None else torch.cat((x, context), dim=1)

        # (2) get q, k, v; dim (batch, seq, inner_dim=dim_head*heads); #! to get k, v: dim => inner_dim*2 => 2 tensor of inner_dim
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # split head and rearrange q k v; d is dim

        # (3) do QK^T/sqrt(d), pre-softmax and mask(optional)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # get QK^T/sqrt(d)
        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # !talking heads, pre-softmax

        if mask is not None:
            mask = repeat(mask, 'b j -> b h i j', h=h, i=n)  
            dots = dots.masked_fill(mask == False, float('-inf'))  # mask attn for padded keys

        # (4) do softmax and dropout and post-softmax
        attn = self.attend(dots)
        attn = self.dropout(attn)
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # !talking heads, post-softmax

        # (5) get out; dim (batch, seq, inner_dim=dim_head*heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # (6) do Linear; dim (batch, seq, inner_dim=dim_head*heads) => dim (batch, seq, dim)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, ffn_ratio, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout), depth=ind + 1),
                LayerScale(dim, FeedForward(dim, ffn_ratio, dropout=dropout), depth=ind + 1)
            ]))
    def forward(self, x, context=None, mask=None):

        for attn, ff in self.layers:  # here attn 
            x = attn(x, context=context, mask=mask) + x  
            x = ff(x) + x
        return x


class SimpleTransformerVAE(nn.Module):
    def __init__(
        self,
        input_dims, #! if N groups, then a tuple of N integers
        num_classes,
        dim,  # model dim 
        dim_latent,  # dim of latent space
        depth_enc,  # depth of encoder
        depth_dec,  # depth of decoder
        heads,  # heads number
        dim_head,  # dim of each head
        ffn_ratio,  # ratio of inner/in dimension in FeedForward
        dropout,  
    ):
        super().__init__()
        self.n_colls = len(input_dims)
        self.token_emb = nn.ModuleList()
        self.token_unembed = nn.ModuleList()
        for input_dim in input_dims:
            self.token_emb.append(
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, dim),
                    nn.GELU(),
                    nn.LayerNorm(dim),
                    nn.Linear(dim, 4 * dim),
                    nn.GELU(),
                    nn.LayerNorm(4 * dim),
                    nn.Linear(4 * dim, dim),
                )
            )
            self.token_unembed.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim), 
                    nn.GELU(),
                    nn.LayerNorm(dim),
                    nn.Linear(dim, 4 * dim),
                    nn.GELU(),
                    nn.LayerNorm(4 * dim),
                    nn.Linear(4 * dim, input_dim),
                )
            )

        self.encoder = Transformer(dim, depth_enc, heads, dim_head, ffn_ratio, dropout)
        self.decoder = Transformer(dim, depth_dec, heads, dim_head, ffn_ratio, dropout)

        self.fc_mu = nn.Linear(dim, dim_latent)  
        self.fc_var = nn.Linear(dim, dim_latent)
        self.fc_z = nn.Linear(dim_latent, dim)

    def encode(self, x_colls):
        '''
            x: embeded features for each Transformer token; dim (batch, seq, dim)
            mask: (batch, dim)
        '''
        # (1) token embedding
        #! for each data_i in (evt, jet, lep, pho), do: (batch, seq_i, dim_i) ==> (batch, seq_i, dim)
        x = [self.token_emb[ind](x_colls[ind]) for ind in range(self.n_colls)] 
        #! concate (evt, jet, lep, pho) data at seq dimension
        x = torch.cat(x, dim=1) # dim: (batch, seq, dim)

        # (2) pass encoder 
        # ! (batch, seq, dim) ==> (batch, seq, dim)
        x = self.encoder(x)

        # (3) Split the result into mu and log(var) components
        mu = self.fc_mu(x)  #! (batch, seq, dim) ==> (batch, seq, dim_latent)
        log_var = self.fc_var(x)  #! (batch, seq, dim) ==> (batch, seq, dim_latent)

        return mu, log_var

    def decode(self, z, seq_lens):
        '''
            x: decodede input features for each Transformer token; dim (batch, seq, dim)
            mask: (batch, dim)
        '''
        #! (batch, seq, dim_latent) ==> (batch, seq, dim)
        z = self.fc_z(z)  
        #! (batch, seq, dim) ==> (batch, seq, dim)
        x_out = self.decoder(z)  
        
        # token unembedding
        #!(batch, seq, dim) ==> ((batch, seq_1, dim), ..., (batch, seq_4, dim))
        x_out_colls = torch.split(x_out, seq_lens, dim=1)
        #! for each data_i in (evt, jet, lep, pho), do: (batch, seq_i, dim) ==> (batch, seq_i, dim_i)
        x_out_colls = [self.token_unembed[ind](x_out_colls[ind]) for ind in range(self.n_colls)]
        
        return x_out_colls

    def reparameterize(self, mu, logvar):
        # ???????????????????
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # a random term
        return eps * std + mu

    def forward(self, *args):
        '''
            args: list of 3*N tensors, i.e. (features, vectors, masks) for N groups
                feature: shape = (batch, input_dim, seq)
                vector: shape = (batch, 4, seq)
                mask: shape = (batch, 1, seq)
        '''
        
        assert len(args) == 3 * self.n_colls
        x_colls = [args[ind * 3].transpose(1, 2) for ind in range(self.n_colls)] #! a list of tensors for each input group, dim: (batch, seq, input_dim)
        seq_lens = [x_colls[ind].shape[1] for ind in range(self.n_colls)]  # list of length of seq for N features
        
        mask_colls = [args[ind * 3 + 2].squeeze(1).bool() for ind in range(self.n_colls)] #! used to compute loss; dim: (batch, seq)

        mu, log_var = self.encode(x_colls)
        z = self.reparameterize(mu, log_var)
        x_out_colls = self.decode(z, seq_lens)
        
        return x_out_colls, x_colls, mask_colls, mu, log_var


class VAELoss(torch.nn.Module):
    '''
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    '''

    def __init__(self, kld_weight=1.):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, recons_colls, input_colls, mask_colls, mu, log_var):
        '''
            recons_colls, input_colls, and mask_colls: list of corresponding tensors for each input group.
                recons_colls: (batch, seq, input_dim) for each element
                input_colls: (batch, seq, input_dim) ...
                mask_colls: (batch, seq) ...
            mu, log_var: in dim (batch, seq, dim_latent)
        '''

        # both losses need to omit masked tokens
        #! mask_colls[inds]: dim (batch, seq_i); 0/1
        
        # do mean on batch 
        def mse_loss(A, B):  
            squared_error = (A - B) ** 2
            return torch.sum(torch.mean(squared_error, dim=0))
            
        loss_recons = [mse_loss(recons_colls[inds] * mask_colls[inds].unsqueeze(-1), \
            input_colls[inds] * mask_colls[inds].unsqueeze(-1)) for inds in range(len(recons_colls))]
        loss_recons = sum(loss_recons)

        mask = torch.cat(mask_colls, dim=1)  # ! [(batch, seq_1), ..., (batch, seq_4)] ==> (batch, seq)
        loss_kld = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1) # (batch, seq)
        loss_kld = loss_kld.masked_fill(~mask, 0).sum(dim=1).mean(dim=0)  #! do mean on batch dimensiom; sum on (seq, dim)

        loss = loss_recons + self.kld_weight * loss_kld
        return {'loss': loss, 'loss_recons': loss_recons.detach(), 'loss_kld': -loss_kld.detach()}



# Weaver inferface

from weaver.utils.logger import _logger
def get_model(data_config, **kwargs):
    # (1) set model config
    cfg = dict(
        input_dims=tuple([len(data_config.input_dicts[k]) for k in ['evt_features', 'jet_features', 'lep_features', 'pho_features']]),
        num_classes=len(data_config.label_value),
        dim=64,
        dim_latent=64,
        depth_enc=4,
        depth_dec=4,
        heads=4,
        dim_head=64,
        ffn_ratio=4,
        dropout=0.1,
    )
    
    kwargs.pop('kld_weight', None)  # remove kld_weight(if exist) 
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    # (2) model instantiation
    model = SimpleTransformerVAE(**cfg)

    # (3) this is the weaver utility requried to export onnx models, ignore
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info

def get_loss(data_config, **kwargs):
    kld_weight = kwargs.get('kld_weight', 1.0)  # get kld_weight from **kwargs, if not exist, get kld_weight=1
    
    return VAELoss(kld_weight=kld_weight)


