# =============================================================================
# Libs
# =============================================================================
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re

from custom_models.transformers.vit.model import PatchEmbedding
from custom_models.transformers.vit.model import TransformerEncoderBlock
# =============================================================================
# Transformer
# =============================================================================

class AttentionHead(nn.Module):
    """
    One head of the self-attention layer
    """

    def __init__(self, head_size, num_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        # tril is a lower triangular matrix. it is not a parameter
        # of the model, so we assign it to the module using register_buffer
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # let's also add dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # Tril matrix (lower triagular matrix) is used to mask 
        # future positions (setting them to -inf) so that the
        # decoder "learns" to predict next words
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        # weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out

def attention(q, k, v, mask = None, dropout = None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])
    
    #mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)
    
    scores = F.softmax(scores, dim = -1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim=None,block_size=None, head_size=None,num_embed=None,dropout=0.1,algorithm="BERT"):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.algorithm = algorithm
        if self.algorithm == "BERT":
            self.linear = nn.Linear(out_dim, out_dim*3)

            self.n_heads = n_heads
            self.out_dim = out_dim
            self.out_dim_per_head = out_dim // n_heads
            self.out = nn.Linear(out_dim, out_dim)
        
        elif self.algorithm == "GPT":
            # num_heads, head_size, num_embed, block_size, dropout
            self.heads = nn.ModuleList(
                [
                    AttentionHead(
                        head_size=head_size,
                        num_embed=num_embed,
                        block_size=block_size,
                        dropout=dropout,
                    )
                    for _ in range(n_heads)
                ]
            )
            self.proj = nn.Linear(num_embed, num_embed)
    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)
    
    def forward(self, x, y=None, mask=None):
        if self.algorithm == "BERT":
            #in decoder, y comes from encoder. In encoder, y=x
            y = x if y is None else y
            
            qkv = self.linear(x) # BS * SEQ_LEN * (3*EMBED_SIZE_L)
            q = qkv[:, :, :self.out_dim] # BS * SEQ_LEN * EMBED_SIZE_L
            k = qkv[:, :, self.out_dim:self.out_dim*2] # BS * SEQ_LEN * EMBED_SIZE_L
            v = qkv[:, :, self.out_dim*2:] # BS * SEQ_LEN * EMBED_SIZE_L
            
            #break into n_heads
            q, k, v = [self.split_heads(t) for t in (q,k,v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
            q, k, v = [t.transpose(1,2) for t in (q,k,v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
            
            #n_heads => attention => merge the heads => mix information
            scores = attention(q, k, v, mask, self.dropout) # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
            scores = scores.transpose(1,2).contiguous().view(scores.shape[0], -1, self.out_dim) # BS * SEQ_LEN * EMBED_SIZE_L
            out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE
            
        elif self.algorithm == "GPT":
            # output of the self-attention
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            # apply the linear projection layer
            out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, inp_dim=None, inner_dim=None,num_embed=None, dropout=0.1,algorithm="BERT"):
        super().__init__()
        self.algorithm = algorithm
        if self.algorithm == "BERT":
            # self.linear1 = nn.Linear(inp_dim, inner_dim)
            # self.linear2 = nn.Linear(inner_dim, inp_dim)
            # self.dropout = nn.Dropout(dropout)
            self.net = nn.Sequential(
                # in the Attention is All You Need paper
                # authors are using the size of the ffwd layer 2048
                # and the output of the model is 512
                # so we apply the same factor of 4
                nn.Linear(inp_dim, inner_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                # apply the linear projection layer
                nn.Linear(inner_dim, inp_dim),
                
            )
            
        elif self.algorithm == "GPT":
            self.net = nn.Sequential(
                # in the Attention is All You Need paper
                # authors are using the size of the ffwd layer 2048
                # and the output of the model is 512
                # so we apply the same factor of 4
                nn.Linear(num_embed, 4 * num_embed),
                nn.ReLU(),
                # apply the linear projection layer
                nn.Linear(4 * num_embed, num_embed),
                nn.Dropout(dropout),
            )
    
    def forward(self, x):
        #inp => inner => relu => dropout => inner => inp
        return self.net(x)
        #return self.linear2(self.dropout(F.relu(self.linear1(x)))) 

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        # n_heads, out_dim=None,block_size=None, head_size=None,num_embed=None,dropout=0.1,algorithm="BERT")
        self.mha = MultiHeadAttention(n_heads=n_heads, out_dim=inner_transformer_size,block_size=None, head_size=None,num_embed=None,dropout=dropout,algorithm="BERT")
        # (self, inp_dim=None, inner_dim=None,num_embed=None, dropout=0.1,algorithm="BERT")
        # inp_dim, inner_dim, dropout=0.1)
        self.ff = FeedForward(inp_dim=inner_transformer_size, inner_dim=inner_ff_size,num_embed=None, dropout=dropout,algorithm="BERT")
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class TransformerBlock(nn.Module):
    """
    This calss will group together MultiHead Attention and
    FeedForward NN, so that we can copy it in Transformer
    """

    def __init__(self, num_heads, block_size, num_embed, dropout):
        super().__init__()
        head_size = num_embed // num_heads
        self.sa = MultiHeadAttention(
            n_heads=num_heads,
            out_dim=None,
            head_size=head_size,
            num_embed=num_embed,
            block_size=block_size,
            dropout=dropout,
            algorithm="GPT"
        )
        self.ffwd = FeedForward(inp_dim=None,inner_dim=None,num_embed=num_embed, dropout=dropout,algorithm="GPT")
        # add the layer normalization
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)

    def forward(self, x):
        # "x +" is the skip (or residual) connection
        # it helps with optimization
        # also we apply layer normalization before self-attention
        # and feed-forward (a reshufle from original paper)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, n_code=None,
                 n_heads=4,
                 block_size=8,
                 num_layers=4, 
                 embed_size=32, 
                 inner_ff_size=None
                , n_embeddings=None,
                seq_len=None,
                vocab_size=100, 
                img_size:int=224, # Training resolution from Table 3 in ViT paper
                in_channels:int=3, # Number of channels in input image
                embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                attn_dropout:float=0, # Dropout for attention projection
                mlp_dropout:float=0.1, # Dropout for dense/MLP layers 
                embedding_dropout:float=0.1, # Dropout for patch and position embeddings,
                num_classes:int=1000, # Default for ImageNet but can customize this
                dropout=0.1,
                algorithm="BERT"):
    
        # N_heads = 12 for VIT
        # block_size == patch_size = 16 for vit
        # num_layers = 12 for VIT:
        super().__init__()
        
        self.algorithm = algorithm
        if self.algorithm == "BERT":
            #model input
            self.embeddings = nn.Embedding(n_embeddings, embed_size)
            self.pe = PositionalEmbedding(embed_size, seq_len)
            
            #backbone
            encoders = []
            for i in range(n_code):
                encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
            self.encoders = nn.ModuleList(encoders)
            
            #language model
            self.norm = nn.LayerNorm(embed_size)
            self.linear = nn.Linear(embed_size, n_embeddings, bias=False)
        elif self.algorithm == "GPT":
            # a simple lookup table that stores embeddings of a fixed dictionary and size
            # each token directly reads off the logits for the next token from a lookup table
            # see more: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            self.vocab_size = vocab_size
            self.num_embed = embed_size
            self.block_size = block_size
            self.num_heads = n_heads
            self.num_layers = num_layers
            self.dropout = dropout
            # each token reads the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(self.vocab_size, self.num_embed)
            # each position from 0 to block_size-1 will get its embedding
            self.position_embedding_table = nn.Embedding(self.block_size, self.num_embed)
            self.blocks = nn.Sequential(
                *[
                    TransformerBlock(
                        num_heads=self.num_heads,
                        block_size=self.block_size,
                        num_embed=self.num_embed,
                        dropout=self.dropout,
                    )
                    for _ in range(self.num_layers)
                ]
            )
            # we add the layer norm before the Linear layer
            self.ln_f = nn.LayerNorm(self.num_embed)
            self.lm_head = nn.Linear(self.num_embed, self.vocab_size)
            
        elif self.algorithm == "VIT":
            # 3. Make the image size is divisble by the patch size 
            assert img_size % block_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {block_size}."
            
            # 4. Calculate number of patches (height * width/patch^2)
            self.num_patches = (img_size * img_size) // block_size**2
                    
            # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
            self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                                requires_grad=True)
            
            # 6. Create learnable position embedding
            self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                                requires_grad=True)
                    
            # 7. Create embedding dropout value
            self.embedding_dropout = nn.Dropout(p=embedding_dropout)
            
            # 8. Create patch embedding layer
            self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                                patch_size=block_size,
                                                embedding_dim=embedding_dim)
            
            # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential()) 
            # Note: The "*" means "all"
            self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                                num_heads=n_heads,
                                                                                mlp_size=mlp_size,
                                                                                mlp_dropout=mlp_dropout) for _ in range(num_layers)])
        
            # 10. Create classifier head
            self.classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape=embedding_dim),
                nn.Linear(in_features=embedding_dim, 
                        out_features=num_classes)
            )
    
    def forward(self, x,targets=None):
        if self.algorithm == "BERT":
            x = self.embeddings(x)
            x = x + self.pe(x)
            for encoder in self.encoders:
                x = encoder(x)
            x = self.norm(x)
            x = self.linear(x)
            return x
        
        elif self.algorithm == "GPT":
            B, T = x.shape
            # idx and targets are (B,T) tensor of integers
            # the token_emb is (B, T, C), C = NUM_EMBED
            token_emb = self.token_embedding_table(x)
            # (T, C)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            posit_emb = self.position_embedding_table(torch.arange(T, device=device))

            x = token_emb + posit_emb
            # apply one head of self-attention
            x = self.blocks(x)
            # (B, T, vocab_size)
            logits = self.lm_head(x)
            # compute the loss
            if targets != None:
                # cross_entropy accepts inputs in a (batch_size, num_classes)
                # so we need to reformat our logits dimensions to
                # (batch_size * time, dim_vocabulary), time = block_size
                B, T, C = logits.shape
                logits = torch.reshape(logits, (B * T, C))
                targets = torch.reshape(targets, (B * T,))
                loss = F.cross_entropy(logits, targets)
            else:
                loss = None
            return logits, loss
        
        elif self.algorithm == "VIT":
            # 12. Get batch size
            batch_size = x.shape[0]
            
            # 13. Create class token embedding and expand it to match the batch size (equation 1)
            class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

            # 14. Create patch embedding (equation 1)
            x = self.patch_embedding(x)

            # 15. Concat class embedding and patch embedding (equation 1)
            x = torch.cat((class_token, x), dim=1)

            # 16. Add position embedding to patch embedding (equation 1) 
            x = self.position_embedding + x

            # 17. Run embedding dropout (Appendix B.1)
            x = self.embedding_dropout(x)

            # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
            x = self.transformer_encoder(x)

            # 19. Put 0 index logit through classifier (equation 4)
            x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

            return x   

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context too the  last block_size tokens
            # because tokens don't communicate between blocks
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution with probabilities probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:,:x.size(1)] #x.size(1) = seq_len