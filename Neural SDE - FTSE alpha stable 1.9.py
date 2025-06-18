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
from torch.utils.data import Dataset, DataLoader

###############################################################################
# 1) Ensure Reproducibility
###############################################################################
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
seed_everything(SEED)

###############################################################################
# 2) Device Setup
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

###############################################################################
# 3) Load & Preprocess CSV (first differences)
###############################################################################
def load_data(path):
    df = pd.read_excel(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    prices = df['Open'].values
    dates  = df['Date'].values
    # compute first differences (d_t = p_t - p_{t-1})
    diffs = pd.Series(prices).diff().dropna().values
    return prices, dates[1:], diffs  # diffs aligned to dates[1:]

###############################################################################
# 4) Create Sliding Windows on diffs
###############################################################################
def create_sliding_windows(series, input_window, forecast_horizon, step_size):
    X, Y, idxs = [], [], []
    n = len(series)
    for start in range(0, n - input_window - forecast_horizon + 1, step_size):
        end = start + input_window
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
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data_tensor, times_torch)
    return data_tensor, coeffs

###############################################################################
# 6) Dataset & DataLoader
###############################################################################
class ForecastDataset(Dataset):
    def __init__(self, data_tensor, coeffs, targets):
        self.data_tensor = data_tensor
        self.coeffs = coeffs
        self.targets = targets
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return (
            self.data_tensor[idx],
            self.coeffs[idx],
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

###############################################################################
# 7) Neural SDE with α-stable Lévy Motion
###############################################################################
class LipSwish(nn.Module):
    def forward(self, x): return 0.909 * torch.nn.functional.silu(x)

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim, num_layers):
        super().__init__()
        act = LipSwish()
        layers = [nn.Linear(in_size, hidden_dim), act]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers.append(nn.Linear(hidden_dim, out_size))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class NeuralSDEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.sde_type, self.noise_type = 'ito', 'diagonal'
        self.linear_in = nn.Linear(1 + hidden_dim + input_dim, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers)
        self.noise_in = nn.Linear(1 + hidden_dim + input_dim, hidden_dim)
        self.g_net = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers)
    def set_X(self, coeffs, times):
        self.X = torchcde.CubicSpline(coeffs, times)
    def f(self, t, y):
        x_t = self.X.evaluate(t)
        t_t = torch.full((y.size(0),1), t, device=y.device)
        inp = torch.cat([t_t, y, x_t], dim=-1)
        return self.f_net(self.linear_in(inp))
    def g(self, t, y):
        x_t = self.X.evaluate(t)
        t_t = torch.full((y.size(0),1), t, device=y.device)
        inp = torch.cat([t_t, y, x_t], dim=-1)
        return self.g_net(self.noise_in(inp))

class LevySDEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, alpha=1.9, beta=0.0):
        super().__init__()
        self.func = NeuralSDEFunc(input_dim, hidden_dim, num_layers)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.alpha, self.beta = alpha, beta

    def sample_levy(self, shape, device):
        U = (torch.rand(shape, device=device) - 0.5) * math.pi
        W = torch.empty(shape, device=device).exponential_(1.0)
        const = self.beta * math.tan(math.pi * self.alpha / 2)
        phi = math.atan(const) / self.alpha
        S = (1 + const**2)**(1/(2*self.alpha))
        num = torch.sin(self.alpha * (U + phi))
        den = torch.cos(U)**(1/self.alpha)
        frac = (torch.cos(U - self.alpha*(U+phi)) / W)**((1-self.alpha)/self.alpha)
        return S * num / den * frac

    def forward(self, coeffs, times):
        self.func.set_X(coeffs, times)
        y = self.initial(self.func.X.evaluate(times[0]))
        ys = [y]
        for i in range(1, len(times)):
            t_prev = times[i-1]; dt = times[i] - t_prev
            drift = self.func.f(t_prev, y) * dt
            dL = self.sample_levy((y.size(0),1), y.device)
            jump = self.func.g(t_prev, y) * dL
            y = y + drift + jump
            ys.append(y)
        z = torch.stack(ys, dim=0).permute(1,0,2)
        return self.decoder(z)

###############################################################################
# 8) Main: Train/Val/Test on diffs, reconstruct prices, no leakage
###############################################################################
def main():
    prices, dates, diffs = load_data("FTSE UK rel.xlsx")
    N = len(diffs)

    # 60/20/20 split on diffs
    train_end = int(0.6 * N)
    val_end   = int(0.8 * N)

    # standardize diffs on train only
    mean_train = diffs[:train_end].mean()
    std_train  = diffs[:train_end].std() + 1e-8
    diffs_std  = (diffs - mean_train) / std_train

    # windowing
    inp_win, fh, step = 30, 4, 2
    X, Y, idxs = create_sliding_windows(diffs_std, inp_win, fh, step)

    # split indices
    train_idx = [i for i,(s,e,te) in enumerate(idxs) if te <= train_end]
    val_idx   = [i for i,(s,e,te) in enumerate(idxs) if s >= train_end and te <= val_end]
    test_idx  = [i for i,(s,e,te) in enumerate(idxs) if s >= val_end]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]
    idxs_test = [idxs[i] for i in test_idx]

    # build CDE data
    tr_data, tr_coef = build_cde_data_from_input(X_train)
    vl_data, vl_coef = build_cde_data_from_input(X_val)
    ts_data, ts_coef = build_cde_data_from_input(X_test)

    # loaders
    batch_size = 16
    train_loader = DataLoader(ForecastDataset(tr_data,tr_coef,Y_train), batch_size, shuffle=True)
    val_loader   = DataLoader(ForecastDataset(vl_data,vl_coef,Y_val),   batch_size)
    test_loader  = DataLoader(ForecastDataset(ts_data,ts_coef,Y_test),  batch_size)

    # model, optimizer, loss
    model = LevySDEModel(2,16,1,2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # time grid
    dt = 1/(inp_win-1)
    times = torch.linspace(0, 1+dt*fh, inp_win+fh).to(device)

    # early stopping on val
    best_val, no_imp = float('inf'), 0
    patience = 3
    for epoch in range(1, 101):
        model.train()
        train_loss = 0
        for _, coeff, targ in train_loader:
            coeff, targ = coeff.to(device), targ.to(device)
            optimizer.zero_grad()
            pred = model(coeff, times).squeeze(-1)[:, inp_win:]
            loss = criterion(pred, targ)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, coeff, targ in val_loader:
                coeff, targ = coeff.to(device), targ.to(device)
                pred = model(coeff, times).squeeze(-1)[:, inp_win:]
                val_loss += criterion(pred, targ).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: Train={train_loss:.6f} | Val={val_loss:.6f}")
        if val_loss < best_val:
            best_val, no_imp = val_loss, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # final test evaluation & reconstruction
    pred_map, true_map = {h:[] for h in range(1,fh+1)}, {h:[] for h in range(1,fh+1)}
    model.eval()
    with torch.no_grad():
        for (s,e,_), coeff, targ in zip(idxs_test, ts_coef, Y_test):
            coeff = coeff.unsqueeze(0).to(device)
            pred_std = model(coeff, times).squeeze(0).squeeze(-1)[inp_win:].cpu().numpy()

            # reconstruct absolute prices for predictions
            anchor = prices[e]
            rec, cp = [], anchor
            for d in pred_std:
                cp += d * std_train + mean_train
                rec.append(cp)

            # reconstruct absolute prices for true horizons
            true_abs = []
            cp_true = anchor
            for d in targ:
                cp_true += d.item() * std_train + mean_train
                true_abs.append(cp_true)

            for h in range(fh):
                pred_map[h+1].append(rec[h])
                true_map[h+1].append(true_abs[h])

    # ----- NEW: build dates_horiz exactly as before, but adapted for "dates" and "idxs_test" -----
    # 'dates' was returned as dates[1:], so dates[i] corresponds to the original price index (i+1).
    # If e is the "anchor" price index, then the predicted horizon h corresponds to price index (e + h),
    # and its date lives at dates[(e + h) - 1].
    dates_arr = dates  # length == len(diffs) == len(prices)-1
    dates_horiz = {h: [] for h in range(1, fh+1)}
    for (s, e, t_end) in idxs_test:
        for h in range(1, fh+1):
            # map price index (e + h) → dates_arr[(e+h)-1]
            dates_horiz[h].append(dates_arr[e + h - 1])

    # aggregate, plot, metrics
    for h in range(1, fh+1):
        agg_p = np.array(pred_map[h])
        agg_t = np.array(true_map[h])
        n = len(agg_t)

        # 1) Full Out‐of‐Sample with nine percentile ticks
        percentiles = np.linspace(0.1, 0.9, 9)
        tick_idxs_full = [int(math.floor(pct * (n - 1))) for pct in percentiles]
        tick_dates_full = [pd.to_datetime(dates_horiz[h][i]) for i in tick_idxs_full]
        tick_labels_full = [d.strftime("%Y-%m-%d %H:%M") for d in tick_dates_full]

        plt.figure(figsize=(10,4))
        x_full = np.arange(n)
        plt.plot(x_full, agg_t, label='Actual')
        plt.plot(x_full, agg_p, '--', label='Predicted')
        plt.title(f"Levy SDE t+{h} Full Out-of-Sample")
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
        plt.savefig(f"levy_t+{h}_full.png")
        plt.close()

        # 2) Last 100 (or fewer) with nine percentile ticks
        last_n = min(100, n)
        percentiles_last = np.linspace(0.1, 0.9, 9)
        tick_idxs_last = [int(math.floor(pct * (last_n - 1))) for pct in percentiles_last]
        # slice of the last_n dates:
        last_dates = dates_horiz[h][-last_n:]
        tick_dates_last = [pd.to_datetime(last_dates[i]) for i in tick_idxs_last]
        tick_labels_last = [d.strftime("%Y-%m-%d %H:%M") for d in tick_dates_last]

        plt.figure(figsize=(10,4))
        x_last = np.arange(last_n)
        plt.plot(x_last,    agg_t[-last_n:], marker='o', label='Actual')
        plt.plot(x_last,    agg_p[-last_n:], marker='x', label='Predicted')
        plt.title(f"Levy SDE t+{h} Last {last_n}")
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
        plt.savefig(f"levy_t+{h}_last{last_n}.png")
        plt.close()

        # metrics
        e = agg_p - agg_t
        mse = np.mean(e**2)
        mae = np.mean(np.abs(e))
        rmse = math.sqrt(mse)
        mape = np.mean(np.abs(e/agg_t)) * 100 if len(agg_t) > 0 else float('nan')
        dpa = np.mean(np.sign(np.diff(agg_t)) == np.sign(np.diff(agg_p))) * 100 if len(agg_t) > 1 else float('nan')
        print(f"[t+{h}] MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, DPA={dpa:.2f}%")

if __name__ == "__main__":
    main()
