import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# ATTENTION
# -----------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, hidden*2)
        energy = torch.tanh(self.attn(x))
        weights = torch.softmax(self.v(energy), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


# -----------------------
# BiGRU + Attention
# -----------------------
class BiGRU_Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.attention = Attention(hidden_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.layer_norm(out)
        context = self.attention(out)
        return context


# -----------------------
# MLP (FLAT FEATURES)
# -----------------------
class DeepMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# -----------------------
# FINAL MODEL
# -----------------------
class AdvancedStockModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_stocks):
        super().__init__()

        self.temporal = BiGRU_Attention(input_size, hidden_size)
        self.mlp = DeepMLP(input_size)

        # stock embedding
        self.embedding = nn.Embedding(num_stocks, 16)

        # fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 🔥 DUAL HEADS
        self.reg_head = nn.Linear(32, 1)   # regression
        self.cls_head = nn.Linear(32, 1)   # classification

        self._init_weights()

    # -----------------------
    # WEIGHT INIT (IMPORTANT)
    # -----------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------
    # FORWARD
    # -----------------------
    def forward(self, x_seq, x_flat, stock_id):

        temporal_features = self.temporal(x_seq)   # (batch, hidden*2)
        mlp_features = self.mlp(x_flat)            # (batch, 32)
        emb = self.embedding(stock_id)             # (batch, 16)

        combined = torch.cat([temporal_features, mlp_features, emb], dim=1)

        features = self.fusion(combined)

        reg_out = self.reg_head(features)
        cls_out = self.cls_head(features)

        return reg_out, cls_out