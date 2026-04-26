import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from uWNcWN2 import UnconditionalTCN, ConditionalTCNSeparate


# =========================================================
# 1. Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 2. Lorenz system generation
# =========================================================
def generate_lorenz(
    n_steps=1600,
    dt=0.01,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    x0=0.0,
    y0=1.0,
    z0=1.05,
):
    """
    Generate Lorenz system using Euler discretization.

    Returns
    -------
    data : np.ndarray, shape (n_steps, 3)
        Columns are [X, Y, Z].
    """
    x = np.zeros(n_steps, dtype=np.float32)
    y = np.zeros(n_steps, dtype=np.float32)
    z = np.zeros(n_steps, dtype=np.float32)

    x[0], y[0], z[0] = x0, y0, z0

    for t in range(n_steps - 1):
        dx = sigma * (y[t] - x[t])
        dy = x[t] * (rho - z[t]) - y[t]
        dz = x[t] * y[t] - beta * z[t]

        x[t + 1] = x[t] + dt * dx
        y[t + 1] = y[t] + dt * dy
        z[t + 1] = z[t] + dt * dz

    return np.column_stack([x, y, z]).astype(np.float32)


# =========================================================
# 3. Standardization
# =========================================================
def fit_standardizer(train_data):
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean, std


def transform_standardizer(data, mean, std):
    return (data - mean) / std


def inverse_transform_column(x_std, mean_col, std_col):
    return x_std * std_col + mean_col


# =========================================================
# 4. Rolling-window dataset builders
# =========================================================
def build_unconditional_dataset(series, target_col, start_t, end_t, window):
    """
    Build unconditional dataset.

    For each target time t:
        input  = target_series[t-window : t]
        target = target_series[t]

    Parameters
    ----------
    series : np.ndarray, shape (N, 3)
    target_col : int
    start_t : int
    end_t : int
    window : int

    Returns
    -------
    X : np.ndarray, shape (num_samples, window, 1)
    y : np.ndarray, shape (num_samples, 1)
    """
    X_list = []
    y_list = []

    target_series = series[:, target_col]

    for t in range(start_t, end_t + 1):
        x_win = target_series[t - window:t].reshape(window, 1)
        y_t = np.array([target_series[t]], dtype=np.float32)

        X_list.append(x_win)
        y_list.append(y_t)

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    return X, y


def build_conditional_dataset(series, target_col, start_t, end_t, window):
    """
    Build conditional dataset.

    Main input:
        target coordinate history
    Conditions:
        the other two coordinates

    Returns
    -------
    X_main : np.ndarray, shape (num_samples, window, 1)
    c_list : list[np.ndarray]
        two arrays, each shape (num_samples, window, 1)
    y : np.ndarray, shape (num_samples, 1)
    """
    X_main_list = []
    C1_list = []
    C2_list = []
    y_list = []

    all_cols = [0, 1, 2]
    cond_cols = [c for c in all_cols if c != target_col]

    main_series = series[:, target_col]
    cond1_series = series[:, cond_cols[0]]
    cond2_series = series[:, cond_cols[1]]

    for t in range(start_t, end_t + 1):
        x_win = main_series[t - window:t].reshape(window, 1)
        c1_win = cond1_series[t - window:t].reshape(window, 1)
        c2_win = cond2_series[t - window:t].reshape(window, 1)
        y_t = np.array([main_series[t]], dtype=np.float32)

        X_main_list.append(x_win)
        C1_list.append(c1_win)
        C2_list.append(c2_win)
        y_list.append(y_t)

    X_main = np.stack(X_main_list).astype(np.float32)
    C1 = np.stack(C1_list).astype(np.float32)
    C2 = np.stack(C2_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)

    return X_main, [C1, C2], y


# =========================================================
# 5. Train / predict / metric
# =========================================================
def train_unconditional(
    model,
    X_train,
    y_train,
    n_iter=5000,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
):
    model.to(device)
    model.train()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    loss_hist = []

    for _ in range(n_iter):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.item())

    return loss_hist


def train_conditional(
    model,
    X_train,
    c_train_list,
    y_train,
    n_iter=5000,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
):
    model.to(device)
    model.train()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    c_train_list_t = [torch.tensor(c, dtype=torch.float32, device=device) for c in c_train_list]
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    loss_hist = []

    for _ in range(n_iter):
        optimizer.zero_grad()
        pred = model(X_train_t, c_train_list_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.item())

    return loss_hist


@torch.no_grad()
def predict_unconditional(model, X, device="cpu"):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    pred = model(X_t).cpu().numpy()
    return pred


@torch.no_grad()
def predict_conditional(model, X, c_list, device="cpu"):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    c_list_t = [torch.tensor(c, dtype=torch.float32, device=device) for c in c_list]
    pred = model(X_t, c_list_t).cpu().numpy()
    return pred


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# =========================================================
# 6. Single-coordinate experiment
# =========================================================
def run_one_coordinate_experiment(
    series_std,
    coord_name,
    target_col,
    mean,
    std,
    window=16,
    train_end=999,
    test_end=1500,
    channels=8,
    kernel_size=2,
    cond_kernel_size=2,
    num_layers=4,
    dropout=0.0,
    n_iter=5000,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
):
    # ---------- datasets ----------
    X_train_u, y_train_u = build_unconditional_dataset(
        series_std, target_col=target_col, start_t=window, end_t=train_end, window=window
    )
    X_test_u, y_test_u = build_unconditional_dataset(
        series_std, target_col=target_col, start_t=train_end + 1, end_t=test_end, window=window
    )

    X_train_c, c_train_list, y_train_c = build_conditional_dataset(
        series_std, target_col=target_col, start_t=window, end_t=train_end, window=window
    )
    X_test_c, c_test_list, y_test_c = build_conditional_dataset(
        series_std, target_col=target_col, start_t=train_end + 1, end_t=test_end, window=window
    )

    # ---------- models ----------
    model_u = UnconditionalTCN(
        input_dim=1,
        channels=channels,
        kernel_size=kernel_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    model_c = ConditionalTCNSeparate(
        input_dim=1,
        num_conditions=2,
        channels=channels,
        kernel_size=kernel_size,
        cond_kernel_size=cond_kernel_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    # ---------- training ----------
    loss_u = train_unconditional(
        model_u,
        X_train_u,
        y_train_u,
        n_iter=n_iter,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    loss_c = train_conditional(
        model_c,
        X_train_c,
        c_train_list,
        y_train_c,
        n_iter=n_iter,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    # ---------- prediction ----------
    pred_u = predict_unconditional(model_u, X_test_u, device=device)
    pred_c = predict_conditional(model_c, X_test_c, c_test_list, device=device)

    # standardized scale
    rmse_u_std = rmse(y_test_u, pred_u)
    rmse_c_std = rmse(y_test_c, pred_c)

    # original scale
    mean_col = mean[0, target_col]
    std_col = std[0, target_col]

    y_test_orig = inverse_transform_column(y_test_u, mean_col, std_col)
    pred_u_orig = inverse_transform_column(pred_u, mean_col, std_col)
    pred_c_orig = inverse_transform_column(pred_c, mean_col, std_col)

    rmse_u_orig = rmse(y_test_orig, pred_u_orig)
    rmse_c_orig = rmse(y_test_orig, pred_c_orig)

    return {
        "coord_name": coord_name,
        "y_test_std": y_test_u,
        "pred_u_std": pred_u,
        "pred_c_std": pred_c,
        "rmse_u_std": rmse_u_std,
        "rmse_c_std": rmse_c_std,
        "y_test_orig": y_test_orig,
        "pred_u_orig": pred_u_orig,
        "pred_c_orig": pred_c_orig,
        "rmse_u_orig": rmse_u_orig,
        "rmse_c_orig": rmse_c_orig,
        "loss_u": loss_u,
        "loss_c": loss_c,
    }


# =========================================================
# 7. Plot and save
# =========================================================
def plot_and_save_coordinate_result(result, save_dir, horizon_plot=100):
    coord_name = result["coord_name"]

    y_true = result["y_test_orig"].reshape(-1)
    pred_u = result["pred_u_orig"].reshape(-1)
    pred_c = result["pred_c_orig"].reshape(-1)
    loss_u = result["loss_u"]
    loss_c = result["loss_c"]

    err_u = pred_u - y_true
    err_c = pred_c - y_true

    n = min(horizon_plot, len(y_true))
    t = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(t, y_true[:n], label="True data")
    axes[0, 0].plot(t, pred_u[:n], label="uWN")
    axes[0, 0].set_title(f"{coord_name}: True vs uWN")
    axes[0, 0].set_xlabel("Time step")
    axes[0, 0].legend()

    axes[0, 1].plot(t, y_true[:n], label="True data")
    axes[0, 1].plot(t, pred_c[:n], label="cWN")
    axes[0, 1].set_title(f"{coord_name}: True vs cWN")
    axes[0, 1].set_xlabel("Time step")
    axes[0, 1].legend()

    axes[1, 0].plot(loss_u, label="uWN training loss")
    axes[1, 0].plot(loss_c, label="cWN training loss")
    axes[1, 0].set_title("Training loss")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].legend()

    axes[1, 1].hist(err_u, bins=40, alpha=0.6, label="uWN error")
    axes[1, 1].hist(err_c, bins=40, alpha=0.6, label="cWN error")
    axes[1, 1].set_title("Forecast error histogram")
    axes[1, 1].legend()

    plt.tight_layout()

    file_path = os.path.join(save_dir, f"lorenz_{coord_name}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {file_path}")


# =========================================================
# 8. Main
# =========================================================
def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # save folder
    save_dir = os.path.join(os.getcwd(), "result", "lorenz_figure")
    os.makedirs(save_dir, exist_ok=True)
    print("Figure save directory:", save_dir)

    # hyperparameters
    kernel_size = 2
    cond_kernel_size = 2
    num_layers = 4
    channels = 16
    lr = 1e-3
    weight_decay = 1e-4
    n_iter = 5000
    window = 32
    train_end = 999
    test_end = 1500
    dropout = 0.0

    # data
    data = generate_lorenz(
        n_steps=test_end + 1,
        dt=0.01,
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        x0=0.0,
        y0=1.0,
        z0=1.05,
    )

    # standardize using train part only
    mean, std = fit_standardizer(data[:train_end + 1])
    data_std = transform_standardizer(data, mean, std)

    coord_names = ["X", "Y", "Z"]
    results = []

    for target_col, coord_name in enumerate(coord_names):
        print(f"\nRunning experiment for {coord_name}...")

        result = run_one_coordinate_experiment(
            series_std=data_std,
            coord_name=coord_name,
            target_col=target_col,
            mean=mean,
            std=std,
            window=window,
            train_end=train_end,
            test_end=test_end,
            channels=channels,
            kernel_size=kernel_size,
            cond_kernel_size=cond_kernel_size,
            num_layers=num_layers,
            dropout=dropout,
            n_iter=n_iter,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        results.append(result)

        print(
            f"{coord_name}: "
            f"RMSE uWN (std) = {result['rmse_u_std']:.6f}, "
            f"RMSE cWN (std) = {result['rmse_c_std']:.6f}, "
            f"RMSE uWN (orig) = {result['rmse_u_orig']:.6f}, "
            f"RMSE cWN (orig) = {result['rmse_c_orig']:.6f}"
        )

        plot_and_save_coordinate_result(result, save_dir=save_dir, horizon_plot=100)

    summary_df = pd.DataFrame({
        "Coordinate": [r["coord_name"] for r in results],
        "RMSE_uWN_std": [r["rmse_u_std"] for r in results],
        "RMSE_cWN_std": [r["rmse_c_std"] for r in results],
        "RMSE_uWN_orig": [r["rmse_u_orig"] for r in results],
        "RMSE_cWN_orig": [r["rmse_c_orig"] for r in results],
    })

    summary_path = os.path.join(save_dir, "rmse_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved RMSE summary to:", summary_path)
    print("\n===== RMSE Summary =====")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()