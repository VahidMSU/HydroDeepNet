import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_length, dropout_prob=0.2):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projection layers for Query, Key, and Value
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)  # Bias-free
        self.fc_out = nn.Linear(d_model, d_model, bias=False)  # Bias-free
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout_prob)
        self.proj_dropout = nn.Dropout(dropout_prob)
        
        # Layer Normalization (removing batch normalization)
        self.norm = nn.LayerNorm(d_model)
        
        # PReLU activation
        self.activation = nn.PReLU()
    
    def forward(self, x):
        # Apply layer normalization before attention
        x = self.norm(x)
        
        batch_size, seq_length, d_model = x.shape
        assert d_model == self.num_heads * self.head_dim, f"Expected d_model={self.num_heads * self.head_dim}, got {d_model}"

        # Compute query, key, and value matrices
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        # Multiply attention weights with values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)

        # Apply the final projection layer and PReLU activation
        out = self.fc_out(attn_output)
        out = self.proj_dropout(self.activation(out))

        return out
