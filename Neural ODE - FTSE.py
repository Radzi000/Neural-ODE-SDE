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
from torchdiffeq import odeint  # For Neural ODE integration

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
# 3) Load & Preprocess Data
###############################################################################
def load_data(path):
    df = pd.read_excel(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    return df

###############################################################################
# 4) Sliding Windows for Forecasting
###############################################################################
def create_sliding_windows_forecast(series, input_window, forecast_horizon, step_size):
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
# 5) Build CDE Spline Coefficients
###############################################################################
def build_cde_data_from_input(seqs):
    batch, length = seqs.shape
    data = [seq.reshape(length,1) for seq in seqs]
    data_tensor = torch.tensor(np.stack(data, axis=0), dtype=torch.float32)
    times = torch.linspace(0,1,length)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data_tensor, times)
    return data_tensor, coeffs

###############################################################################
# 6) Dataset & DataLoader
###############################################################################
class SP500ForecastDataset(Dataset):
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
# 7) Neural ODE Components
###############################################################################
class LipSwish(nn.Module):
    def forward(self, x): return 0.909 * torch.nn.functional.silu(x)

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim, num_layers):
        super().__init__()
        act = LipSwish()
        layers = [nn.Linear(in_size, hidden_dim), act]
        for _ in range(num_layers-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers.append(nn.Linear(hidden_dim, out_size))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class NeuralODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers):
        super().__init__()
        self.linear_in = nn.Linear(1 + hidden_dim + input_dim, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers)
    def set_X(self, coeffs, times):
        self.X = torchcde.CubicSpline(coeffs, times)
        self.times = times
    def forward(self, t, y):
        x_t = self.X.evaluate(t)
        t_t = torch.full((y.size(0),1), float(t), device=y.device)
        inp = torch.cat([t_t, y, x_t], dim=-1)
        return self.f_net(self.linear_in(inp))

class NeuralODEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.func = NeuralODEFunc(input_dim, hidden_dim, hidden_dim, num_layers)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
    def forward(self, coeffs, times):
        self.func.set_X(coeffs, times)
        y0_raw = self.func.X.evaluate(times[0])
        y0 = self.initial(y0_raw)
        z = odeint(self.func, y0, times, method='dopri5')
        z = z.permute(1,0,2)
        return self.decoder(z)

###############################################################################
# 8) Main: Train/Val/Test Split, Training, Forecasting & Metrics
###############################################################################
def main():
    # load
    df = load_data("FTSE UK rel.xlsx")
    prices_raw = df['Open'].values
    dates_raw  = df['Date'].values

    # differenced series for modeling
    diffs = pd.Series(prices_raw).diff().dropna().values

    # split points for 60/20/20
    N = len(diffs)
    cut1 = int(0.6 * N)
    cut2 = int(0.8 * N)

    # standardize on train only
    mean_train = diffs[:cut1].mean()
    std_train  = diffs[:cut1].std() + 1e-8
    diffs_std  = (diffs - mean_train) / std_train

    # window parameters
    input_window, forecast_horizon, step_size = 30, 4, 2

    # sliding windows on standardized diffs
    X, Y, idxs = create_sliding_windows_forecast(diffs_std, input_window, forecast_horizon, step_size)

    # create masks
    train_mask = np.array([end <= cut1 for (_,end,_) in idxs])
    val_mask   = np.array([(start >= cut1) and (end <= cut2) for (start,end,_) in idxs])
    test_mask  = np.array([ start >= cut2 for (start,_,_) in idxs])

    X_tr, Y_tr = X[train_mask], Y[train_mask]
    X_vl, Y_vl = X[val_mask],   Y[val_mask]
    X_ts, Y_ts = X[test_mask],  Y[test_mask]
    idxs_ts     = [idxs[i] for i in np.where(test_mask)[0]]

    print(f"Train windows: {len(X_tr)} | Val windows: {len(X_vl)} | Test windows: {len(X_ts)}")

    # build CDE data for each
    tr_data, tr_coef = build_cde_data_from_input(X_tr)
    vl_data, vl_coef = build_cde_data_from_input(X_vl)
    ts_data, ts_coef = build_cde_data_from_input(X_ts)

    # DataLoaders
    bs = 16
    tr_ld = DataLoader(SP500ForecastDataset(tr_data, tr_coef, Y_tr), bs, shuffle=True)
    vl_ld = DataLoader(SP500ForecastDataset(vl_data, vl_coef, Y_vl), bs)
    ts_ld = DataLoader(SP500ForecastDataset(ts_data, ts_coef, Y_ts), bs)

    # model, optimizer, loss
    model = NeuralODEModel(input_dim=1, hidden_dim=16, output_dim=1, num_layers=2).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # time grid
    dt    = 1/(input_window-1)
    times = torch.linspace(0, 1+dt*forecast_horizon, input_window+forecast_horizon).to(device)

    # training with early stopping on validation
    best_val, wait, patience = float('inf'), 0, 3
    for ep in range(1, 51):
        model.train()
        tr_loss = 0
        for _, coef, targ in tr_ld:
            coef, targ = coef.to(device), targ.to(device)
            opt.zero_grad()
            out_full = model(coef, times).squeeze(-1)
            out = out_full[:, input_window:]
            loss = loss_fn(out, targ)
            loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        model.eval()
        vl_loss = 0
        with torch.no_grad():
            for _, coef, targ in vl_ld:
                coef, targ = coef.to(device), targ.to(device)
                out_full = model(coef, times).squeeze(-1)
                out = out_full[:, input_window:]
                vl_loss += loss_fn(out, targ).item()
        vl_loss /= len(vl_ld)

        print(f"Epoch {ep}: Train={tr_loss:.6f} | Val={vl_loss:.6f}")
        if vl_loss < best_val:
            best_val, wait = vl_loss, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep}")
                break

    # final evaluation on test set
    pred_map, true_map = {h:[] for h in range(1,forecast_horizon+1)}, {h:[] for h in range(1,forecast_horizon+1)}
    model.eval()
    with torch.no_grad():
        for (start,end,_), coef in zip(idxs_ts, ts_coef):
            coef = coef.unsqueeze(0).to(device)
            pred_std = model(coef, times).squeeze(0).squeeze(-1)[input_window:].cpu().numpy()
            pred_diff = pred_std * std_train + mean_train
            anchor = prices_raw[end]
            rec = []
            cur = anchor
            for d in pred_diff:
                cur += d
                rec.append(cur)
            true_vals = prices_raw[end+1:end+1+forecast_horizon]
            for h in range(1,forecast_horizon+1):
                pred_map[h].append(rec[h-1])
                true_map[h].append(true_vals[h-1])

    # --------------- Build date-lists for plotting ---------------
    dates_arr = dates_raw
    dates_horiz = {h: [] for h in range(1, forecast_horizon+1)}
    for h in range(1, forecast_horizon+1):
        for (s, e, _) in idxs_ts:
            dates_horiz[h].append(dates_arr[e + h])

    # --------------- Plotting: Full-series + Last-100 with percentiles ---------------
    for h in range(1, forecast_horizon+1):
        P = np.array(pred_map[h])
        T = np.array(true_map[h])
        n = len(P)

        # Full Out-of-Sample with date ticks
        percentiles = np.linspace(0.1, 0.9, 9)
        tick_idxs_full = [int(math.floor(pct * (n - 1))) for pct in percentiles]
        tick_dates_full = [pd.to_datetime(dates_horiz[h][i]) for i in tick_idxs_full]
        tick_labels_full = [d.strftime("%Y-%m-%d %H:%M") for d in tick_dates_full]

        plt.figure(figsize=(10, 5))
        x_full = np.arange(n)
        plt.plot(x_full, T, label='Actual')
        plt.plot(x_full, P, '--', label='Predicted')
        plt.title(f"t+{h} Full Out-of-Sample")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.xticks(tick_idxs_full, tick_labels_full, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"t+{h}_full.png")
        plt.close()

        # Last 100 points
        last_n = min(100, n)
        percentiles_last = np.linspace(0.1, 0.9, 9)
        tick_idxs_last = [int(math.floor(pct * (last_n - 1))) for pct in percentiles_last]
        date_slice_last = dates_horiz[h][-last_n:]
        tick_dates_last = [pd.to_datetime(date_slice_last[i]) for i in tick_idxs_last]
        tick_labels_last = [d.strftime("%Y-%m-%d %H:%M") for d in tick_dates_last]

        plt.figure(figsize=(10, 5))
        x_last = np.arange(last_n)
        plt.plot(x_last,    T[-last_n:], marker='o', label='Actual')
        plt.plot(x_last,    P[-last_n:], marker='x', label='Predicted')
        plt.title(f"t+{h} Last {last_n}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.xticks(tick_idxs_last, tick_labels_last, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"t+{h}_last{last_n}.png")
        plt.close()

        # Metrics
        e = P - T
        mse = np.mean(e**2)
        mae = np.mean(np.abs(e))
        rmse = math.sqrt(mse)
        mape = np.mean(np.abs(e/T))*100 if len(T)>0 else float('nan')
        dpa  = np.mean(np.sign(np.diff(T)) == np.sign(np.diff(P))) * 100 if len(T)>1 else float('nan')
        print(f"[t+{h}] MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, DPA={dpa:.2f}%")

if __name__ == "__main__":
    main()
