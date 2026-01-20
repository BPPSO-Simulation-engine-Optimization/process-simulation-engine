import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm_a = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_b = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout_a = nn.Dropout(dropout)
        self.dropout_b = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention: query=x, key=x, value=x
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout_a(attn_output)
        out_a = self.layernorm_a(x + attn_output)
        
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output)
        return self.layernorm_b(out_a + ffn_output)

class ProcessTransformer(nn.Module):
    def __init__(self, num_activities, max_len=50, embed_dim=36, num_heads=4, ff_dim=64, dropout=0.1):
        """
        Process Transformer model for next activity and duration prediction.
        
        Args:
            num_activities: Size of the vocabulary (number of unique activities). 
                            Typically includes padding index (0) and activity encodings.
            max_len: Maximum length of input sequences.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward dimension in Transformer block.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_activities, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.dense_1 = nn.Linear(embed_dim, 64)
        
        # Heads
        # Activity: Classification over num_activities
        self.activity_head = nn.Linear(64, num_activities) 
        # Duration: Regression (scalar)
        self.duration_head = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch, seq_len] of activity indices
        seq_len = x.size(1)
        device = x.device
        
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0) # [1, seq_len]
        
        x_emb = self.embedding(x) + self.pos_embedding(positions) # [batch, seq_len, embed_dim]
        
        x_trans = self.transformer_block(x_emb) # [batch, seq_len, embed_dim]
        
        # Global Average Pooling (mean over sequence length)
        # Note: Masking padding would be ideal but simple GlobalAvg is standard in basic implementations
        x_pool = x_trans.mean(dim=1) # [batch, embed_dim]
        
        x_hidden = self.dropout(x_pool)
        x_hidden = torch.relu(self.dense_1(x_hidden))
        x_hidden = self.dropout(x_hidden)
        
        next_activity = self.activity_head(x_hidden)
        next_duration = self.duration_head(x_hidden)
        
        return next_activity, next_duration
