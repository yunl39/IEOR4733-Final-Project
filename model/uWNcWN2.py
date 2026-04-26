import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. Basic causal conv
# =========================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


# =========================================================
# 2. uWN block
# =========================================================
class UWNBlock(nn.Module):
    """
    Standard unconditional dilated residual block.
    """
    def __init__(self, channels, kernel_size, dilation, dropout=0.0):
        super().__init__()

        self.hidden_conv = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation
        )

        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

        self.residual_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_1x1 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, h):
        z = self.hidden_conv(h)
        z = self.act(z)
        z = self.dropout(z)

        residual = self.residual_1x1(z)
        skip = self.skip_1x1(z)

        h_next = h + residual
        return h_next, skip


# =========================================================
# 3. cWN block
# =========================================================
class CWNBlock(nn.Module):
    """
    Conditional dilated residual block:
        z = conv_x(h) + sum_j conv_cj(c_j)
    """
    def __init__(
        self,
        channels,
        num_conditions,
        kernel_size,
        cond_kernel_size,
        dilation,
        dropout=0.0
    ):
        super().__init__()

        self.hidden_conv = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation
        )

        self.cond_convs = nn.ModuleList([
            CausalConv1d(
                in_channels=1,
                out_channels=channels,
                kernel_size=cond_kernel_size,
                dilation=dilation
            )
            for _ in range(num_conditions)
        ])

        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

        self.residual_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_1x1 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, h, c_list):
        z = self.hidden_conv(h)

        for conv_c, c in zip(self.cond_convs, c_list):
            z = z + conv_c(c)

        z = self.act(z)
        z = self.dropout(z)

        residual = self.residual_1x1(z)
        skip = self.skip_1x1(z)

        h_next = h + residual
        return h_next, skip


# =========================================================
# 4. uWN model
# =========================================================
class UnconditionalTCN(nn.Module):
    def __init__(
        self,
        input_dim,
        channels=8,
        kernel_size=2,
        num_layers=4,
        dropout=0.0
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            UWNBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        self.final_act = nn.LeakyReLU(0.1)
        self.final_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (batch, T, input_dim)
        x = x.transpose(1, 2)      # (batch, input_dim, T)
        h = self.input_proj(x)     # (batch, channels, T)

        skip_sum = None
        for block in self.blocks:
            h, skip = block(h)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        out = self.final_act(skip_sum)
        out = self.final_1x1(out)

        out_last = out[:, :, -1]
        y_hat = self.fc(out_last)
        return y_hat


# =========================================================
# 5. cWN model
# =========================================================
class ConditionalTCNSeparate(nn.Module):
    def __init__(
        self,
        input_dim,
        num_conditions,
        channels=8,
        kernel_size=2,
        cond_kernel_size=2,
        num_layers=4,
        dropout=0.0
    ):
        super().__init__()

        self.num_conditions = num_conditions
        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            CWNBlock(
                channels=channels,
                num_conditions=num_conditions,
                kernel_size=kernel_size,
                cond_kernel_size=cond_kernel_size,
                dilation=2 ** i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        self.final_act = nn.LeakyReLU(0.1)
        self.final_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x, c_list):
        if len(c_list) != self.num_conditions:
            raise ValueError(
                f"Expected {self.num_conditions} condition series, got {len(c_list)}."
            )

        x = x.transpose(1, 2)          # (batch, input_dim, T)
        h = self.input_proj(x)         # (batch, channels, T)

        c_list = [c.transpose(1, 2) for c in c_list]   # each -> (batch, 1, T)

        skip_sum = None
        for block in self.blocks:
            h, skip = block(h, c_list)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        out = self.final_act(skip_sum)
        out = self.final_1x1(out)

        out_last = out[:, :, -1]
        y_hat = self.fc(out_last)
        return y_hat