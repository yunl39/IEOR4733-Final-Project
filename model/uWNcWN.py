import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Basic causal conv
# ─────────────────────────────────────────────
class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with left padding only.

    Input shape:
        (batch, in_channels, length)
    Output shape:
        (batch, out_channels, length)
    """
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


# ─────────────────────────────────────────────
# 2. Unconditional residual block
# ─────────────────────────────────────────────
class ResidualBlockUnconditional(nn.Module):
    """
    One unconditional residual block:
        x -> dilated causal conv -> ReLU -> dropout
          -> residual 1x1 conv -> add to x
          -> skip 1x1 conv for final aggregation
    """
    def __init__(self, channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.dilated_conv = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_1x1 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        z = self.dilated_conv(x)
        z = self.relu(z)
        z = self.dropout(z)

        residual = self.residual_1x1(z)
        skip = self.skip_1x1(z)

        out = x + residual
        return out, skip


# ─────────────────────────────────────────────
# 3. Conditional residual block: shared condition conv
# ─────────────────────────────────────────────
class ResidualBlockConditionalShared(nn.Module):
    """
    Condition model where all condition series are stacked as channels
    and processed by one shared condition convolution.

    x branch: conv_x(x)
    c branch: conv_c(c)
    then add -> ReLU -> dropout -> residual/skip
    """
    def __init__(self, channels, cond_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.conv_x = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.conv_c = CausalConv1d(
            in_channels=cond_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_1x1 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, c):
        zx = self.conv_x(x)
        zc = self.conv_c(c)

        z = zx + zc
        z = self.relu(z)
        z = self.dropout(z)

        residual = self.residual_1x1(z)
        skip = self.skip_1x1(z)

        out = x + residual
        return out, skip


# ─────────────────────────────────────────────
# 4. Conditional residual block: separate conv per condition
# ─────────────────────────────────────────────
class ResidualBlockConditionalSeparate(nn.Module):
    """
    Each condition series gets its own extra convolution.

    x branch: conv_x(x)
    each condition c_j branch: conv_cj(c_j)

    z = conv_x(x) + sum_j conv_cj(c_j)
    """
    def __init__(self, channels, num_conditions, kernel_size, dilation, dropout=0.0):
        super().__init__()

        self.conv_x = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation
        )

        self.cond_convs = nn.ModuleList([
            CausalConv1d(
                in_channels=1,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation
            )
            for _ in range(num_conditions)
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_1x1 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, c_list):
        z = self.conv_x(x)

        for conv_c, c in zip(self.cond_convs, c_list):
            z = z + conv_c(c)

        z = self.relu(z)
        z = self.dropout(z)

        residual = self.residual_1x1(z)
        skip = self.skip_1x1(z)

        out = x + residual
        return out, skip


# ─────────────────────────────────────────────
# 5. Unconditional model
# ─────────────────────────────────────────────
class UnconditionalTCN(nn.Module):
    """
    Input:
        x: (batch, T, input_dim)

    Output:
        y_hat: (batch, 1)
    """
    def __init__(
        self,
        input_dim,
        channels=32,
        kernel_size=2,
        num_layers=4,
        dropout=0.0
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            ResidualBlockUnconditional(
                channels=channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        self.output_relu = nn.ReLU()
        self.output_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (batch, T, input_dim)
        x = x.transpose(1, 2)          # -> (batch, input_dim, T)
        h = self.input_proj(x)         # -> (batch, channels, T)

        skip_sum = None
        for block in self.blocks:
            h, skip = block(h)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        h = self.output_relu(skip_sum)
        h = self.output_1x1(h)         # (batch, channels, T)

        h_last = h[:, :, -1]           # (batch, channels)
        y_hat = self.fc(h_last)        # (batch, 1)
        return y_hat


# ─────────────────────────────────────────────
# 6. Conditional model: shared condition conv
# ─────────────────────────────────────────────
class ConditionalTCNShared(nn.Module):
    """
    Input:
        x: (batch, T, input_dim)
        c: (batch, T, cond_dim)

    Output:
        y_hat: (batch, 1)
    """
    def __init__(
        self,
        input_dim,
        cond_dim,
        channels=32,
        kernel_size=2,
        num_layers=4,
        dropout=0.0
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.cond_proj = nn.Conv1d(cond_dim, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            ResidualBlockConditionalShared(
                channels=channels,
                cond_channels=channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        self.output_relu = nn.ReLU()
        self.output_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x, c):
        # x: (batch, T, input_dim)
        # c: (batch, T, cond_dim)
        x = x.transpose(1, 2)          # -> (batch, input_dim, T)
        c = c.transpose(1, 2)          # -> (batch, cond_dim, T)

        h = self.input_proj(x)         # -> (batch, channels, T)
        c = self.cond_proj(c)          # -> (batch, channels, T)

        skip_sum = None
        for block in self.blocks:
            h, skip = block(h, c)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        h = self.output_relu(skip_sum)
        h = self.output_1x1(h)

        h_last = h[:, :, -1]
        y_hat = self.fc(h_last)
        return y_hat


# ─────────────────────────────────────────────
# 7. Conditional model: separate conv per condition
# ─────────────────────────────────────────────
class ConditionalTCNSeparate(nn.Module):
    """
    Input:
        x: (batch, T, input_dim)
        c_list: list of condition tensors
            each c_j has shape (batch, T, 1)

    Output:
        y_hat: (batch, 1)
    """
    def __init__(
        self,
        input_dim,
        num_conditions,
        channels=32,
        kernel_size=2,
        num_layers=4,
        dropout=0.0
    ):
        super().__init__()

        self.num_conditions = num_conditions
        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            ResidualBlockConditionalSeparate(
                channels=channels,
                num_conditions=num_conditions,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        self.output_relu = nn.ReLU()
        self.output_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x, c_list):
        # x: (batch, T, input_dim)
        # c_list: list of length num_conditions, each (batch, T, 1)

        if len(c_list) != self.num_conditions:
            raise ValueError(
                f"Expected {self.num_conditions} condition series, got {len(c_list)}."
            )

        x = x.transpose(1, 2)          # -> (batch, input_dim, T)
        h = self.input_proj(x)         # -> (batch, channels, T)

        c_list = [c.transpose(1, 2) for c in c_list]   # each -> (batch, 1, T)

        skip_sum = None
        for block in self.blocks:
            h, skip = block(h, c_list)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        h = self.output_relu(skip_sum)
        h = self.output_1x1(h)

        h_last = h[:, :, -1]
        y_hat = self.fc(h_last)
        return y_hat


# ─────────────────────────────────────────────
# 8. Example training step
# ─────────────────────────────────────────────
def train_one_step_unconditional(model, optimizer, loss_fn, x, y):
    model.train()
    optimizer.zero_grad()

    y_hat = model(x)
    loss = loss_fn(y_hat, y)

    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_step_cond_shared(model, optimizer, loss_fn, x, c, y):
    model.train()
    optimizer.zero_grad()

    y_hat = model(x, c)
    loss = loss_fn(y_hat, y)

    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_step_cond_separate(model, optimizer, loss_fn, x, c_list, y):
    model.train()
    optimizer.zero_grad()

    y_hat = model(x, c_list)
    loss = loss_fn(y_hat, y)

    loss.backward()
    optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────
# 9. Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    # -----------------------------
    # Example dimensions
    # -----------------------------
    batch_size = 16
    T = 30
    input_dim = 5
    cond_dim = 3
    num_conditions = 3

    # -----------------------------
    # Fake data
    # -----------------------------
    x = torch.randn(batch_size, T, input_dim)      # main input
    c = torch.randn(batch_size, T, cond_dim)       # shared-condition input
    c_list = [
        torch.randn(batch_size, T, 1) for _ in range(num_conditions)
    ]                                               # separate conditions
    y = torch.randn(batch_size, 1)                 # target

    # -----------------------------
    # 1) Unconditional model
    # -----------------------------
    model_u = UnconditionalTCN(
        input_dim=input_dim,
        channels=32,
        kernel_size=2,
        num_layers=4,
        dropout=0.1
    )

    y_hat_u = model_u(x)
    print("Unconditional output shape:", y_hat_u.shape)

    optimizer_u = torch.optim.Adam(model_u.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loss_u = train_one_step_unconditional(model_u, optimizer_u, loss_fn, x,   y)
    print("Unconditional one-step loss:", loss_u)

    # -----------------------------
    # 2) Conditional model (shared)
    # -----------------------------
    model_cs = ConditionalTCNShared(
        input_dim=input_dim,
        cond_dim=cond_dim,
        channels=32,
        kernel_size=2,
        num_layers=4,
        dropout=0.1
    )

    y_hat_cs = model_cs(x, c)
    print("Conditional shared output shape:", y_hat_cs.shape)

    optimizer_cs = torch.optim.Adam(model_cs.parameters(), lr=1e-3)
    loss_cs = train_one_step_cond_shared(model_cs, optimizer_cs, loss_fn, x, c, y)
    print("Conditional shared one-step loss:", loss_cs)

    # -----------------------------
    # 3) Conditional model (separate)
    # -----------------------------
    model_sep = ConditionalTCNSeparate(
        input_dim=input_dim,
        num_conditions=num_conditions,
        channels=32,
        kernel_size=2,
        num_layers=4,
        dropout=0.1
    )

    y_hat_sep = model_sep(x, c_list)
    print("Conditional separate output shape:", y_hat_sep.shape)

    optimizer_sep = torch.optim.Adam(model_sep.parameters(), lr=1e-3)
    loss_sep = train_one_step_cond_separate(
        model_sep, optimizer_sep, loss_fn, x, c_list, y
    )
    print("Conditional separate one-step loss:", loss_sep)