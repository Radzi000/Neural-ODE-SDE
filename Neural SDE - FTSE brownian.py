import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchcde
import torchsde
from torch.utils.data import Dataset, DataLoader

###############################################################################
# 1) Ensure Reproducibility
###############################################################################
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
seed_everything(SEED)

###############################################################################
# 2) Device Setup
###############################################################################
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device:", device)

###############################################################################
# 3) Load & Preprocess CSV
###############################################################################
def load_data(path):
    """
    Now returns:
      - dates:   array of all timestamps (aligned with prices)
      - prices:  numpy array of open prices
      - diffs:   numpy array of first differences of prices (length = len(prices)-1)
    """
    df = pd.read_excel(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)

    dates  = df['Date'].values
    prices = df['Open'].values

    # differences for stationary modeling
    diffs = pd.Series(prices).diff().dropna().values
    return dates, prices, diffs

###############################################################################
# 4) Sliding Windows
###############################################################################
def create_sliding_windows(series, input_window, forecast_horizon, step_size):
    X, Y, idxs = [], [], []
    n = len(series)
    for start in range(0, n - input_window - forecast_horizon + 1, step_size):
        end   = start + input_window
        t_end = end + forecast_horizon
        X.append(series[start:end])
        Y.append(series[end:t_end])
        idxs.append((start, end, t_end))
    return np.array(X), np.array(Y), idxs

###############################################################################
# 5) Build Cubic Spline Data
###############################################################################
def build_cde_data_from_input(seqs):
    batch, length = seqs.shape
    times_np = np.linspace(0, 1, length)
    data = [np.stack([times_np, seq], axis=-1) for seq in seqs]
    data_tensor = torch.tensor(np.array(data, dtype=np.float32))
    times_torch = torch.linspace(0, 1, length)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
        data_tensor, times_torch
    )
    return data_tensor, coeffs

###############################################################################
# 6) Dataset & DataLoader
###############################################################################
class ForecastDataset(Dataset):
    def __init__(self, data_tensor, coeffs, targets):
        self.data_tensor = data_tensor
        self.coeffs      = coeffs
        self.targets     = targets
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return (
            self.data_tensor[idx],
            self.coeffs[idx],
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

###############################################################################
# 7) Neural SDE Model Components
###############################################################################
class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim, num_layers, activation='lipswish'):
        super().__init__()
        act = LipSwish() if activation=='lipswish' else nn.ReLU()
        layers = [nn.Linear(in_size, hidden_dim), act]
        for _ in range(num_layers-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers.append(nn.Linear(hidden_dim, out_size))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class NeuralSDEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers):
        super().__init__()
        self.sde_type, self.noise_type = 'ito', 'diagonal'
        self.linear_in = nn.Linear(1 + hidden_dim + input_dim, hidden_dim)
        self.f_net     = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers)
        self.noise_in  = nn.Linear(1 + hidden_dim + input_dim, hidden_dim)
        self.g_net     = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers)

    def set_X(self, coeffs, times):
        self.X     = torchcde.CubicSpline(coeffs, times)
        self.times = times

    def f(self, t, y):
        xt = self.X.evaluate(t)
        tt = torch.full((y.size(0),1), t, device=y.device)
        inp = torch.cat([tt, y, xt], dim=-1)
        return self.f_net(self.linear_in(inp))

    def g(self, t, y):
        xt = self.X.evaluate(t)
        tt = torch.full((y.size(0),1), t, device=y.device)
        inp = torch.cat([tt, y, xt], dim=-1)
        return self.g_net(self.noise_in(inp))

class NeuralSDEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.func    = NeuralSDEFunc(input_dim, hidden_dim, hidden_dim, num_layers)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, coeffs, times):
        self.func.set_X(coeffs, times)
        y0 = self.initial(self.func.X.evaluate(times[0]))
        dt = times[1] - times[0]
        z = torchsde.sdeint(self.func, y0, times, dt=dt, method='euler')
        z = z.permute(1,0,2)
        return self.decoder(z)

###############################################################################
# 8) Main: Train/Val/Test & Early Stopping on Validation
###############################################################################
def main():
    # --- load dates, prices, diffs
    dates, prices, diffs = load_data("FTSE UK rel.xlsx")
    N = len(diffs)

    # 60/20/20 split
    train_end = int(0.6 * N)
    val_end   = int(0.8 * N)

    # standardize diffs based on training set
    mean_train = diffs[:train_end].mean()
    std_train  = diffs[:train_end].std() + 1e-8
    diffs_std  = (diffs - mean_train) / std_train

    # sliding‐window parameters
    inp_win, fh, step = 30, 4, 2
    X, Y, idxs = create_sliding_windows(diffs_std, inp_win, fh, step)

    # determine train/val/test window indices
    train_idx = [i for i,(s,e,te) in enumerate(idxs) if te <= train_end]
    val_idx   = [i for i,(s,e,te) in enumerate(idxs) if s >= train_end and te <= val_end]
    test_idx  = [i for i,(s,e,te) in enumerate(idxs) if s >= val_end]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]
    idxs_test_filtered = [idxs[i] for i in test_idx]  # list of (s, e, t_end) for test

    # build cubic‐spline coefficients for CDE
    tr_data, tr_coef = build_cde_data_from_input(X_train)
    vl_data, vl_coef = build_cde_data_from_input(X_val)
    ts_data, ts_coef = build_cde_data_from_input(X_test)

    # DataLoaders
    bsize = 16
    tr_loader = DataLoader(ForecastDataset(tr_data, tr_coef, Y_train), bsize, shuffle=True)
    vl_loader = DataLoader(ForecastDataset(vl_data, vl_coef, Y_val),   bsize)
    ts_loader = DataLoader(ForecastDataset(ts_data, ts_coef, Y_test),  bsize)

    # instantiate model, optimizer, loss
    model = NeuralSDEModel(2, 16, 1, 2).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # time grid: from 0 to 1 + dt*fh, with (inp_win + fh) points
    dt    = 1/(inp_win-1)
    times = torch.linspace(0, 1+dt*fh, inp_win+fh).to(device)

    # training loop with early stopping on validation loss
    best_val, wait, pat = float('inf'), 0, 3
    for ep in range(1, 101):
        model.train()
        tl = 0.0
        for _, coef, targ in tr_loader:
            coef, targ = coef.to(device), targ.to(device)
            opt.zero_grad()
            out = model(coef, times).squeeze(-1)[:, inp_win:]
            l   = loss_fn(out, targ)
            l.backward()
            opt.step()
            tl += l.item()
        tl /= len(tr_loader)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for _, coef, targ in vl_loader:
                coef, targ = coef.to(device), targ.to(device)
                out = model(coef, times).squeeze(-1)[:, inp_win:]
                vl += loss_fn(out, targ).item()
        vl /= len(vl_loader)

        print(f"Epoch {ep}: Train={tl:.4f} | Val={vl:.4f}")
        if vl < best_val:
            best_val, wait = vl, 0
        else:
            wait += 1
            if wait >= pat:
                print(f"Early stopping at epoch {ep}")
                break

    # final test evaluation: collect preds & trues for each horizon 1..fh
    preds = {h: [] for h in range(1, fh+1)}
    trues = {h: [] for h in range(1, fh+1)}

    model.eval()
    with torch.no_grad():
        for (s,e,_), coef in zip(idxs_test_filtered, ts_coef):
            # out: shape (1, inp_win+fh, 1) -> squeeze -> (inp_win+fh,)
            pred_std = model(coef.unsqueeze(0).to(device), times).squeeze().cpu().numpy()[inp_win:]
            anchor = prices[e]  # e is the index of last observed price
            rec = []
            cp = anchor
            for d in pred_std:
                # undo standardization
                cp += d*std_train + mean_train
                rec.append(cp)
            true = prices[e+1:e+1+fh]  # the next fh true prices
            for h in range(fh):
                if h < len(rec):
                    preds[h+1].append(rec[h])
                    trues[h+1].append(true[h])

    # 9) Metrics
    for h in range(1, fh+1):
        p = np.array(preds[h])
        t = np.array(trues[h])
        e = p - t
        mse  = np.mean(e**2)
        mae  = np.mean(np.abs(e))
        rmse = math.sqrt(mse)
        mape = np.mean(np.abs(e/t))*100 if len(t)>0 else float('nan')
        dpa  = np.mean(np.sign(np.diff(t)) == np.sign(np.diff(p))) * 100 if len(t)>1 else float('nan')
        print(f"[t+{h}] MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, DPA={dpa:.2f}%")

    # --------------- Build date‐lists for plotting ---------------
    # dates is an array of length len(prices).  For each test window (s,e,_),
    # the predicted price for horizon h corresponds to price index (e + h),
    # so we extract dates[e + h].
    dates_arr = dates  # numpy array of dtype datetime64
    dates_horiz = {h: [] for h in range(1, fh+1)}
    for h in range(1, fh+1):
        for (s, e, te) in idxs_test_filtered:
            dates_horiz[h].append(dates_arr[e + h])

    # 10) Plotting: 4 full‐series + 4 last‐100 with percentile ticks
    for h in range(1, fh+1):
        p = np.array(preds[h])
        t = np.array(trues[h])
        n = len(p)

        # Compute tick positions at 10%, 20%, ..., 90% of the index range
        percentiles = np.linspace(0.1, 0.9, 9)
        tick_idxs_full = [int(math.floor(pct * (n - 1))) for pct in percentiles]
        tick_dates_full = [pd.to_datetime(dates_horiz[h][i]) for i in tick_idxs_full]
        tick_labels_full = [d.strftime("%Y-%m-%d %H:%M") for d in tick_dates_full]

        # --- Full out‐of‐sample (indices on x, but nine ticks labeled by date) ---
        plt.figure(figsize=(10, 5))
        x_full = np.arange(n)
        plt.plot(x_full, t, label='Actual')
        plt.plot(x_full, p, '--', label='Predicted')
        plt.title(f"t+{h} Full Out‐of‐Sample")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)

        plt.xticks(
            tick_idxs_full,
            tick_labels_full,
            rotation=45,
            ha='right'
        )

        plt.tight_layout()
        plt.savefig(f"FTSE_diff_t+{h}_full.png")
        plt.close()

        # 2) Last 100 (or fewer)
        last_n = min(100, n)
        percentiles_last = np.linspace(0.1, 0.9, 9)
        tick_idxs_last = [int(math.floor(pct * (last_n - 1))) for pct in percentiles_last]
        date_slice_last = dates_horiz[h][-last_n:]  # last_n dates
        tick_dates_last = [pd.to_datetime(date_slice_last[i]) for i in tick_idxs_last]
        tick_labels_last = [d.strftime("%Y-%m-%d %H:%M") for d in tick_dates_last]

        plt.figure(figsize=(10, 5))
        x_last = np.arange(last_n)
        plt.plot(x_last,    t[-last_n:], marker='o', label='Actual')
        plt.plot(x_last,    p[-last_n:], marker='x', label='Predicted')
        plt.title(f"t+{h} Last {last_n}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)

        plt.xticks(
            tick_idxs_last,
            tick_labels_last,
            rotation=45,
            ha='right'
        )

        plt.tight_layout()
        plt.savefig(f"FTSE_diff_t+{h}_last100.png")
        plt.close()

if __name__ == "__main__":
    main()
