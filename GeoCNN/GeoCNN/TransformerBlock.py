from GeoCNN.Attention_Mechanism import MultiHeadSelfAttention
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, seq_length=256)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, forward_expansion * d_model),
            nn.PReLU(),
            nn.Linear(forward_expansion * d_model, d_model)
        )
    
    def forward(self, x, verbose=False):
        if verbose:
            print(f"shape of x in TransformerBlock before attention: {x.shape}")
        attn_output = self.attention(x)
        if verbose:
            print(f"shape of attn_output in TransformerBlock: {attn_output.shape}")
        x = self.norm1(attn_output + x)  # Add & Norm
        if verbose:
            print(f"shape of x in TransformerBlock before feed_forward: {x.shape}")
        feed_forward_output = self.feed_forward(x)
        if verbose:
            print(f"shape of feed_forward_output in TransformerBlock: {feed_forward_output.shape}")
        return self.norm2(feed_forward_output + x)

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, forward_expansion):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, forward_expansion) for _ in range(num_layers)]
        )
    
    def forward(self, x, verbose=False):
        for layer in self.layers:
            if verbose:
                print(f"shape of x in TransformerEncoder before layer: {x.shape} in layer: {layer}")
            x = layer(x)
        return x