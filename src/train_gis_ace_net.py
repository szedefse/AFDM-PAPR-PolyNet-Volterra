import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time
import sys

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
# --- System Parameters ---
N = 64  # Total subcarriers (子载波总数，即帧长度)
Q = 8  # Guard Interval length (保护间隔长度，用于消除多径干扰)
D = N - 2 * Q - 1  # Data symbols length (数据符号长度)
M_mod = 4  # QAM order (调制阶数，4对应QPSK，16对应16QAM)
# AFDM Parameters (AFDM 核心参数)
c1 = (2 * 1 + 1) / (2 * N) # Chirp parameter 1 (第一啁啾参数，控制频率分集)
c2 = 1.0 / (N ** 2) # Chirp parameter 2 (第二啁啾参数，控制频率分集)
# Training Hyperparameters (训练超参数)
BATCH_SIZE = 512 # (批大小，每次训练的样本数)
EPOCHS = 150 # (总训练轮数)
BATCHES_PER_EPOCH = 50 # (每轮包含的批次数)
LEARNING_RATE = 0.001 # (学习率)
MASK_THRESHOLD = 0.9 # (遮罩阈值，决定哪些峰值需要被处理)

# --- Model Switches (模型开关) ---
# Set to True to enable, False to disable (设置为True启用，False禁用)
ENABLE_MODELS = {
    "DNN": True,
    "ResNet": True,
    "DenseNet": False,
    "GatedMLP": True,
    "CVNN": False,
    "SIREN": False,
    "SE-ResNet": True,
    "PolyNet-Dense": False,
    "SE-PolyNet": False,
    "PolyNet-Volterra": True,
    "PolyNet-Res": False,
    "PolyNet": True
}


# --- Model Definitions ---
# 1. DNN (Baseline)
class DNN_Baseline(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(DNN_Baseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, x):
        inp = torch.cat([x.real, x.imag], dim=1).float()
        out = self.net(inp)
        return torch.complex(out[:, :64], out[:, 64:])


# 2. ResNet (Residual MLP)
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNet_MLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_blocks=3):
        super(ResNet_MLP, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        inp = torch.cat([x.real, x.imag], dim=1).float()
        x = self.in_proj(inp)
        for block in self.res_blocks:
            x = block(x)
        out = self.out_proj(x)
        return torch.complex(out[:, :64], out[:, 64:])


# 3. DenseNet (Dense MLP)
class DenseNet_MLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_layers=4):
        super(DenseNet_MLP, self).__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ))
            current_dim += hidden_dim
        self.out_layer = nn.Linear(current_dim, 128)

    def forward(self, x):
        inp = torch.cat([x.real, x.imag], dim=1).float()
        features = [inp]
        for layer in self.layers:
            curr_in = torch.cat(features, dim=1)
            out = layer(curr_in)
            features.append(out)
        final_in = torch.cat(features, dim=1)
        out = self.out_layer(final_in)
        return torch.complex(out[:, :64], out[:, 64:])


# 4. Gated MLP
class GatedBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GatedBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.linear(x) * self.sigmoid(self.gate(x))


class Gated_MLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(Gated_MLP, self).__init__()
        self.net = nn.Sequential(
            GatedBlock(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            GatedBlock(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            GatedBlock(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, x):
        inp = torch.cat([x.real, x.imag], dim=1).float()
        out = self.net(inp)
        return torch.complex(out[:, :64], out[:, 64:])


# 5. CVNN (Complex-Valued Linear Layer)
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.real_weight = nn.Parameter(torch.randn(out_features, in_features) / np.sqrt(in_features))
        self.imag_weight = nn.Parameter(torch.randn(out_features, in_features) / np.sqrt(in_features))
        self.real_bias = nn.Parameter(torch.zeros(out_features))
        self.imag_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        xr, xi = x.real, x.imag
        out_r = torch.matmul(xr, self.real_weight.t()) - torch.matmul(xi, self.imag_weight.t()) + self.real_bias
        out_i = torch.matmul(xi, self.real_weight.t()) + torch.matmul(xr, self.imag_weight.t()) + self.imag_bias
        return torch.complex(out_r, out_i)


class CVNN_Model(nn.Module):
    def __init__(self, hidden_dim=64):
        super(CVNN_Model, self).__init__()
        self.layer1 = ComplexLinear(64, hidden_dim)
        self.layer2 = ComplexLinear(hidden_dim, hidden_dim)
        self.layer3 = ComplexLinear(hidden_dim, 64)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = torch.complex(self.act(x.real), self.act(x.imag))
        x = self.layer2(x)
        x = torch.complex(self.act(x.real), self.act(x.imag))
        x = self.layer3(x)
        return x


# 6. SIREN (Sinusoidal Representation Network)
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SIREN_Model(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(SIREN_Model, self).__init__()
        self.net = nn.Sequential(
            SineLayer(input_dim, hidden_dim, is_first=True, omega_0=30),
            SineLayer(hidden_dim, hidden_dim, omega_0=30),
            SineLayer(hidden_dim, hidden_dim, omega_0=30),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, x):
        inp = torch.cat([x.real, x.imag], dim=1).float()
        out = self.net(inp)
        return torch.complex(out[:, :64], out[:, 64:])


# 7. SE-ResNet (Squeeze-and-Excitation ResNet)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.fc(x)
        return x * y


class SEResBlock(nn.Module):
    def __init__(self, dim):
        super(SEResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.se = SEBlock(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = self.se(out)
        out += residual
        return self.relu(out)


class SE_ResNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_blocks=3):
        super(SE_ResNet, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([SEResBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        inp = torch.cat([x.real, x.imag], dim=1).float()
        x = self.in_proj(inp)
        for block in self.blocks:
            x = block(x)
        out = self.out_proj(x)
        return torch.complex(out[:, :64], out[:, 64:])


# 8. PolyNet (Polynomial Expanded Network)
class PolyNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(PolyNet, self).__init__()
        # Expansion: x, y, x^2, y^2, x*y, x^3, y^3 (7 features per subcarrier)
        # Input is full frame N=64.
        self.expanded_dim = 64 * 7
        self.net = nn.Sequential(
            nn.Linear(self.expanded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, x):
        # x is (Batch, 64)
        r = x.real.float()
        i = x.imag.float()
        r2 = r * r
        i2 = i * i
        ri = r * i
        r3 = r2 * r
        i3 = i2 * i

        # Concatenate all features:
        # We want a single vector per batch item.
        # Order: [r_0..r_63, i_0..i_63, r2_0..r2_63, ...]
        inp = torch.cat([r, i, r2, i2, ri, r3, i3], dim=1)
        out = self.net(inp)
        return torch.complex(out[:, :64], out[:, 64:])


# 9. SE-PolyNet (PolyNet with Squeeze-and-Excitation Attention)
class SE_PolyNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, reduction=4):
        super(SE_PolyNet, self).__init__()
        # Expansion: x, y, x^2, y^2, x*y, x^3, y^3
        self.expanded_dim = 64 * 7
        
        # 1. First Projection
        self.fc1 = nn.Linear(self.expanded_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.Tanh()
        
        # 2. SE Attention Block
        self.se_fc1 = nn.Linear(hidden_dim, hidden_dim // reduction, bias=False)
        self.se_relu = nn.ReLU()
        self.se_fc2 = nn.Linear(hidden_dim // reduction, hidden_dim, bias=False)
        self.se_sigmoid = nn.Sigmoid()
        
        # 3. Second Projection
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.Tanh()
        
        # 4. Output
        self.fc_out = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        # --- Polynomial Expansion ---
        r = x.real.float()
        i = x.imag.float()
        r2 = r * r
        i2 = i * i
        ri = r * i
        r3 = r2 * r
        i3 = i2 * i

        inp = torch.cat([r, i, r2, i2, ri, r3, i3], dim=1)
        
        # --- Network Flow ---
        x = self.fc1(inp)
        x = self.ln1(x)
        x = self.act1(x)
        
        # --- SE Attention ---
        w = self.se_fc1(x)
        w = self.se_relu(w)
        w = self.se_fc2(w)
        w = self.se_sigmoid(w)
        
        # Scale:
        x = x * w
        
        # --- Rest of Network ---
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)
        
        out = self.fc_out(x)
        return torch.complex(out[:, :64], out[:, 64:])


# 10. PolyNet-Volterra (Physical Perception - Minimalist 1st+3rd Order)
class PolyNet_Volterra(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(PolyNet_Volterra, self).__init__()
        # Expansion: r, i, r*P, i*P (4 features per subcarrier)
        # Power P = r^2 + i^2
        self.expanded_dim = 64 * 4
        self.net = nn.Sequential(
            nn.Linear(self.expanded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, x):
        r = x.real.float()
        i = x.imag.float()
        
        P = r*r + i*i
        
        r_P = r * P
        i_P = i * P

        inp = torch.cat([r, i, r_P, i_P], dim=1)
        out = self.net(inp)
        return torch.complex(out[:, :64], out[:, 64:])


# 11. PolyNet-Res (Residual Connection)
class PolyNet_Res(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_blocks=3):
        super(PolyNet_Res, self).__init__()
        # Standard Expansion: 7 features
        self.expanded_dim = 64 * 7
        
        self.in_proj = nn.Sequential(
            nn.Linear(self.expanded_dim, hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        r = x.real.float()
        i = x.imag.float()
        r2 = r * r
        i2 = i * i
        ri = r * i
        r3 = r2 * r
        i3 = i2 * i
        inp = torch.cat([r, i, r2, i2, ri, r3, i3], dim=1)

        x = self.in_proj(inp)
        for block in self.res_blocks:
            x = block(x)
        out = self.out_proj(x)
        return torch.complex(out[:, :64], out[:, 64:])


# 12. PolyNet-Dense (Dense Connection)
class PolyNet_Dense(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_layers=4):
        super(PolyNet_Dense, self).__init__()
        # Standard Expansion: 7 features
        self.expanded_dim = 64 * 7
        
        self.layers = nn.ModuleList()
        current_dim = self.expanded_dim
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ))
            current_dim += hidden_dim
        self.out_layer = nn.Linear(current_dim, 128)

    def forward(self, x):
        r = x.real.float()
        i = x.imag.float()
        r2 = r * r
        i2 = i * i
        ri = r * i
        r3 = r2 * r
        i3 = i2 * i
        inp = torch.cat([r, i, r2, i2, ri, r3, i3], dim=1)

        features = [inp]
        for layer in self.layers:
            curr_in = torch.cat(features, dim=1)
            out = layer(curr_in)
            features.append(out)
        final_in = torch.cat(features, dim=1)
        out = self.out_layer(final_in)
        return torch.complex(out[:, :64], out[:, 64:])


# --- Utility Functions ---
def daft_matrix(n, c1_val, c2_val):
    n_idx = torch.arange(n).float().view(-1, 1)
    vec = torch.arange(n).float()
    lam_c1_h = torch.diag(torch.exp(1j * 2 * np.pi * c1_val * (vec ** 2)))
    lam_c2_h = torch.diag(torch.exp(1j * 2 * np.pi * c2_val * (vec ** 2)))
    m_idx = torch.arange(n).float().view(1, -1)
    idft = torch.exp(1j * 2 * np.pi * n_idx * m_idx / n) / np.sqrt(n)
    T = torch.matmul(torch.matmul(lam_c1_h, idft), lam_c2_h)
    return T


TRANSFORM_MATRIX = daft_matrix(N, c1, c2).cfloat()


def gen_channel_matrix(n, taps, l_max, k_max, c1_val):
    chan_coef = (torch.randn(taps) + 1j * torch.randn(taps)) / np.sqrt(2)
    delay_taps = torch.randint(0, l_max, (taps,))
    delay_taps, _ = torch.sort(delay_taps)
    delay_taps = delay_taps - delay_taps.min()
    doppler_taps_int = torch.randint(-k_max, k_max + 1, (taps,))
    doppler_freq = doppler_taps_int.float() / n

    H = torch.zeros((n, n), dtype=torch.cfloat)
    n_vec = torch.arange(n).float()

    for i in range(taps):
        h_i = chan_coef[i]
        l_i = int(delay_taps[i])
        f_i = doppler_freq[i]
        D_i_diag = torch.exp(-1j * 2 * np.pi * f_i * n_vec)
        D_i = torch.diag(D_i_diag)
        G_i_diag = torch.ones(n, dtype=torch.cfloat)
        indices = torch.arange(l_i)
        if len(indices) > 0:
            term = (n ** 2) - 2 * n * (l_i - indices.float())
            G_i_diag[indices] = torch.exp(-1j * 2 * np.pi * c1_val * term)
        G_i = torch.diag(G_i_diag)
        eye = torch.eye(n, dtype=torch.cfloat)
        Pi_li = torch.roll(eye, shifts=l_i, dims=0)
        term_mat = torch.matmul(torch.matmul(G_i, D_i), Pi_li)
        H += h_i * term_mat
    return H


def apply_mask(time_signal, threshold):
    mag = torch.abs(time_signal)
    mask_indices = mag > threshold
    masked_signal = time_signal.clone()

    if not mask_indices.any():
        return masked_signal, mask_indices

    # Nearest Neighbor Strategy
    # Find nearest valid symbol (mag <= threshold) for each invalid symbol within the same frame

    # 1. Expand dimensions for broadcasting: (Batch, N, N)
    # We want to compute distance from each point i to each point j
    x_i = time_signal.unsqueeze(2)  # Source points (Batch, N, 1)
    x_j = time_signal.unsqueeze(1)  # Candidate points (Batch, 1, N)

    # 2. Compute Euclidean distance matrix
    dists = torch.abs(x_i - x_j)  # (Batch, N, N)

    # 3. Mask out invalid candidates (points that are themselves masked)
    # valid_mask: (Batch, N) -> True if point is valid candidate
    valid_mask = ~mask_indices
    # If a column j corresponds to an invalid point, dist to it should be infinity
    # Expand valid_mask to (Batch, 1, N) for broadcasting over rows i
    dists.masked_fill_(~valid_mask.unsqueeze(1), float('inf'))

    # 4. Find index of nearest valid neighbor
    # (Batch, N)
    nearest_indices = torch.argmin(dists, dim=2)

    # 5. Gather replacement values
    replacements = torch.gather(time_signal, 1, nearest_indices)

    # 6. Apply replacements
    # Check if a frame has at least one valid sample to avoid selecting from all-inf row
    has_valid = valid_mask.any(dim=1)  # (Batch,)

    # Case A: Frames with at least one valid sample -> Use Nearest Neighbor
    # We construct a mask for (Batch, N) where pixel is masked AND frame has valid samples
    nn_mask = mask_indices & has_valid.unsqueeze(1)
    if nn_mask.any():
        masked_signal[nn_mask] = replacements[nn_mask]

    # Case B: Frames with NO valid samples -> Fallback to Clipping
    # (Very rare if N is large and threshold is reasonable, but possible)
    clip_mask = mask_indices & (~has_valid.unsqueeze(1))
    if clip_mask.any():
        masked_signal[clip_mask] = threshold * time_signal[clip_mask] / mag[clip_mask]

    return masked_signal, mask_indices


def construct_afdm_frame_basic(x_p, data):
    batch_size = x_p.shape[0]
    device = x_p.device
    zeros_gi = torch.zeros(batch_size, Q).cfloat().to(device)
    x_tilde = torch.cat([x_p, zeros_gi, data, zeros_gi], dim=1)
    T = TRANSFORM_MATRIX.to(device)
    s = torch.matmul(x_tilde, T.T)
    return s, x_tilde


def get_qam_constellation(M):
    """Generate normalized QAM constellation points"""
    if M == 4:
        points = torch.tensor([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], dtype=torch.cfloat)
        return points / np.sqrt(2)
    elif M == 16:
        # Generate 16-QAM points: {-3, -1, 1, 3} + j{-3, -1, 1, 3}
        real = torch.tensor([-3, -1, 1, 3], dtype=torch.float)
        imag = torch.tensor([-3, -1, 1, 3], dtype=torch.float)
        # Meshgrid manually
        r_grid = real.repeat_interleave(4)
        i_grid = imag.repeat(4)
        points = torch.complex(r_grid, i_grid)
        # Normalize to unit power: E[|x|^2] = (1/16) * sum(r^2 + i^2)
        # Average power of unnormalized 16QAM is 10
        return points / np.sqrt(10)
    else:
        raise NotImplementedError(f"M={M} not supported yet")

def generate_data(batch_size, M):
    points = get_qam_constellation(M)
    
    # Random indices
    pilot_idx = np.random.randint(0, M, (batch_size, 1))
    data_idx = np.random.randint(0, M, (batch_size, D))
    
    # Map to points
    # Need to move points to CPU for indexing if indices are numpy, or convert indices to tensor
    # Simple way:
    points_np = points.numpy()
    x_p = torch.from_numpy(points_np[pilot_idx]).cfloat()
    data = torch.from_numpy(points_np[data_idx]).cfloat()
    
    return x_p, data


def qam_demod(y, M):
    points = get_qam_constellation(M).to(y.device)
    
    y_exp = y.unsqueeze(-1)
    p_exp = points.view(1, 1, -1)
    dist = torch.abs(y_exp - p_exp) ** 2
    idx = torch.argmin(dist, dim=-1)
    return points[idx]


def linear_layer_flops(model):
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            flops += 2 * module.in_features * module.out_features
    return flops


def benchmark_inference_latency(model, device, batch_size=1, repeats=2000, warmup=100):
    model.eval()
    x = torch.randn(batch_size, N, dtype=torch.cfloat, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(repeats):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
    frames = repeats * batch_size
    return elapsed * 1000.0 / frames, frames / elapsed


def save_gpu_complexity_latency_table(models, device, save_dir):
    if device.type != "cuda":
        print("Skipping Table 11 GPU latency benchmark because CUDA is not available.")
        return

    import csv

    rows = []
    gpu_name = torch.cuda.get_device_name(0)
    for model_name, model in models.items():
        model = model.to(device)
        single_latency, _ = benchmark_inference_latency(model, device, batch_size=1, repeats=3000, warmup=200)
        batch_latency, throughput = benchmark_inference_latency(model, device, batch_size=256, repeats=120, warmup=30)
        rows.append({
            "Model": model_name,
            "Device": gpu_name,
            "Parameters": sum(p.numel() for p in model.parameters()),
            "Linear_FLOPs_per_frame": linear_layer_flops(model),
            "Single_frame_GPU_latency_ms": single_latency,
            "Batch256_GPU_latency_ms_per_frame": batch_latency,
            "Batch256_throughput_frames_per_s": throughput,
            "Torch": torch.__version__,
        })

    out_path = os.path.join(save_dir, "complexity_latency_gpu.csv")
    with open(out_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            out_row = row.copy()
            out_row["Single_frame_GPU_latency_ms"] = f"{row['Single_frame_GPU_latency_ms']:.6f}"
            out_row["Batch256_GPU_latency_ms_per_frame"] = f"{row['Batch256_GPU_latency_ms_per_frame']:.6f}"
            out_row["Batch256_throughput_frames_per_s"] = f"{row['Batch256_throughput_frames_per_s']:.2f}"
            writer.writerow(out_row)
    print(f"\n=> Table 11 GPU latency benchmark saved to {out_path}")


# --- Main Training and Evaluation Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA is not available. Training-time output will be CPU time, not RTX 3090 GPU time.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_dir = os.path.join(script_dir, "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)

    # Initialize enabled models based on ENABLE_MODELS config
    models = {}
    
    if ENABLE_MODELS["DNN"]: models["DNN"] = DNN_Baseline()
    if ENABLE_MODELS["ResNet"]: models["ResNet"] = ResNet_MLP()
    if ENABLE_MODELS["DenseNet"]: models["DenseNet"] = DenseNet_MLP()
    if ENABLE_MODELS["GatedMLP"]: models["GatedMLP"] = Gated_MLP()
    if ENABLE_MODELS["CVNN"]: models["CVNN"] = CVNN_Model()
    if ENABLE_MODELS["SIREN"]: models["SIREN"] = SIREN_Model()
    if ENABLE_MODELS["SE-ResNet"]: models["SE-ResNet"] = SE_ResNet()
    if ENABLE_MODELS["PolyNet"]: models["PolyNet"] = PolyNet()
    if ENABLE_MODELS["SE-PolyNet"]: models["SE-PolyNet"] = SE_PolyNet()
    if ENABLE_MODELS["PolyNet-Volterra"]: models["PolyNet-Volterra"] = PolyNet_Volterra()
    if ENABLE_MODELS["PolyNet-Res"]: models["PolyNet-Res"] = PolyNet_Res()
    if ENABLE_MODELS["PolyNet-Dense"]: models["PolyNet-Dense"] = PolyNet_Dense()

    if not models:
        print("Warning: No models enabled in ENABLE_MODELS! Please set at least one model to True.")
        return

    if "--benchmark-only" in sys.argv:
        if device.type != "cuda":
            raise RuntimeError("CUDA is required for the RTX 3090 Table 11 benchmark. Please run this in the PyCharm GPU environment.")
        for model_name, model in models.items():
            weight_path = os.path.join(model_save_dir, f"{model_name}.pth")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"Missing trained weight for {model_name}: {weight_path}")
            state = torch.load(weight_path, map_location=device)
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
        save_dir = os.path.join(script_dir, "image")
        os.makedirs(save_dir, exist_ok=True)
        save_gpu_complexity_latency_table(models, device, save_dir)
        return

    results = {}
    loss_history = {}  # 新增：用于记录训练过程中的损失
    grad_norm_history = {} # 新增：记录截断前的原始梯度范数
    SNR_dB_range = range(0, 26, 5)

    results["Masked_NoNet"] = []
    results["Masked_NoNet_Active"] = []
    results["Masked_NoNet_MSE"] = [] # Store Masked MSE for baseline
    baseline_done = False
    
    # Store performance metrics for final summary
    model_metrics = {}
    training_time_records = []

    for model_name, model in models.items():
        print(f"\n=========================================")
        print(f"Training Model: {model_name}")
        print(f"=========================================")

        model = model.to(device)
        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {total_params}")
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        criterion = nn.MSELoss()

        taps = 3;
        l_max = 2;
        k_max = 1

        model_losses = []  # 新增：记录当前模型每个epoch的loss
        model_grad_norms = [] # 新增：记录截断前的最大峰值梯度范数
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_start_time = time.perf_counter()
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            max_epoch_grad_norm = 0.0 # 改为：追踪本轮遇到的最大梯度峰值

            # Using Dynamic SNR for Gradient Stress Testing
            # But training still uses Dynamic SNR for robustness

            for i in range(BATCHES_PER_EPOCH):
                x_p, data = generate_data(BATCH_SIZE, M_mod)
                x_p, data = x_p.to(device), data.to(device)
                s_orig, _ = construct_afdm_frame_basic(x_p, data)
                s_masked, mask_indices = apply_mask(s_orig, MASK_THRESHOLD)

                if not mask_indices.any(): continue

                H = gen_channel_matrix(N, taps, l_max, k_max, c1).to(device)
                r = torch.matmul(s_masked, H.T)

                # Fixed SNR Training (20dB)
                # 使用固定的 20dB 信噪比进行训练
                snr_db = 20.0
                snr_linear = 10 ** (snr_db / 10)
                noise_std = np.sqrt(1 / (2 * snr_linear))
                noise = noise_std * (torch.randn_like(r) + 1j * torch.randn_like(r))
                r_noisy = r + noise

                eye = torch.eye(N).cfloat().to(device)
                H_H = torch.conj(H.T)
                noise_var = 1 / snr_linear
                mmse_mat = torch.matmul(H_H, H) + noise_var * eye
                mmse_inv = torch.inverse(mmse_mat)
                G = torch.matmul(mmse_inv, H_H)
                s_est = torch.matmul(r_noisy, G.T)

                # Full Frame Input: (Batch, N)
                # We feed the entire frame (s_est) into the model
                input_frame = s_est
                target_frame = s_orig

                optimizer.zero_grad()
                reconstructed_frame = model(input_frame)
                
                # MASKED LOSS ONLY
                # We only care about the error at the masked indices
                # Although we output the whole frame, we only backpropagate on the masked positions
                
                # Extract the relevant predictions and targets
                pred_masked = reconstructed_frame[mask_indices]
                target_masked = target_frame[mask_indices]
                
                # If using unsqueeze(1) in model output before, we don't need it now as model outputs (Batch, N)
                # But [mask_indices] flattens it to (Num_Masked_Points,)
                
                loss = criterion(torch.view_as_real(pred_masked), torch.view_as_real(target_masked))
                
                loss.backward()
                
                # 计算并记录截断前的原始梯度范数
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                # 捕捉异常批次的梯度爆炸瞬间（记录最大值）
                if total_norm > max_epoch_grad_norm:
                    max_epoch_grad_norm = total_norm
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_epoch_loss = epoch_loss / BATCHES_PER_EPOCH
            model_losses.append(avg_epoch_loss)
            model_grad_norms.append(max_epoch_grad_norm)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_epoch_loss:.6f}, MaxGradNorm: {max_epoch_grad_norm:.2f}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_elapsed_seconds = time.perf_counter() - train_start_time
        training_time_records.append({
            "Model": model_name,
            "Device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "Epochs": EPOCHS,
            "Batches_per_epoch": BATCHES_PER_EPOCH,
            "Batch_size": BATCH_SIZE,
            "Training_seconds": train_elapsed_seconds,
            "Training_minutes": train_elapsed_seconds / 60.0,
        })
        print(f"Training time for {model_name}: {train_elapsed_seconds:.2f} s ({train_elapsed_seconds / 60.0:.2f} min)")

        loss_history[model_name] = model_losses  # 新增：保存当前模型的损失历史记录
        grad_norm_history[model_name] = model_grad_norms # 保存梯度范数

        # Save Model
        save_path = os.path.join(model_save_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        # Evaluation
        print(f"Evaluating {model_name}...")
        model.eval()
        ber_list = []
        active_ber_list = []
        masked_mse_list = [] # Store Masked MSE for this model
        
        # Pre-generate evaluation data if not already done (done only once for the first model)
        eval_data_path = os.path.join(script_dir, "eval_data.pt")
        if 'eval_data' not in locals():
            if os.path.exists(eval_data_path):
                print(f"Loading pre-generated evaluation data from {eval_data_path}...")
                eval_data = torch.load(eval_data_path, map_location=device)
            else:
                print("Pre-generating evaluation data...")
                eval_data = {}
                batch_size_eval = 1000
                num_realizations = 100
                
                for snr_db in SNR_dB_range:
                    eval_data[snr_db] = []
                    snr_linear = 10 ** (snr_db / 10)
                    noise_std = np.sqrt(1 / (2 * snr_linear))
                    noise_var = 1 / snr_linear
                    
                    for _ in range(num_realizations):
                        # Generate data
                        x_p_test, data_test = generate_data(batch_size_eval, M_mod)
                        x_p_test, data_test = x_p_test.to(device), data_test.to(device)
                        
                        s_orig_tx, _ = construct_afdm_frame_basic(x_p_test, data_test)
                        s_masked_tx, mask_idx = apply_mask(s_orig_tx, MASK_THRESHOLD)
                        
                        # Channel
                        H = gen_channel_matrix(N, taps, l_max, k_max, c1).to(device)
                        r_masked = torch.matmul(s_masked_tx, H.T)
                        
                        # Noise
                        noise = noise_std * (torch.randn_like(r_masked) + 1j * torch.randn_like(r_masked))
                        r_masked += noise
                        
                        # Equalization
                        eye = torch.eye(N).cfloat().to(device)
                        H_H = torch.conj(H.T)
                        mmse_mat = torch.matmul(H_H, H) + noise_var * eye
                        mmse_inv = torch.inverse(mmse_mat)
                        G = torch.matmul(mmse_inv, H_H)
                        s_est_masked = torch.matmul(r_masked, G.T)
                        
                        # --- IDEAL PATH (No Mask) ---
                        # Apply SAME channel and SAME noise to original unmasked signal
                        # To ensure strict fairness in comparison
                        r_ideal = torch.matmul(s_orig_tx, H.T)
                        r_ideal += noise # Same noise realization
                        s_est_ideal = torch.matmul(r_ideal, G.T)
                        
                        # Store everything needed for evaluation
                        eval_data[snr_db].append({
                            's_est_masked': s_est_masked,
                            's_est_ideal': s_est_ideal, # Stored for Ideal Baseline
                            's_orig_tx': s_orig_tx,
                            'mask_idx': mask_idx,
                            'data_test': data_test,
                            'TRANSFORM_MATRIX': TRANSFORM_MATRIX.to(device)
                        })
                
                print(f"Saving evaluation data to {eval_data_path}...")
                torch.save(eval_data, eval_data_path)

        if not baseline_done:
            ber_no_net_list = []
            active_ber_no_net_list = []
            masked_mse_no_net_list = []
            
            ber_ideal_list = [] # Store Ideal BER
            active_ber_ideal_list = []
            masked_mse_ideal_list = []

        # MSE accumulators for this model (across all SNRs)
        total_mse_sum = 0
        total_mse_count = 0

        for snr_db in SNR_dB_range:
            err_net = 0
            err_no_net = 0
            err_ideal = 0
            total_sym = 0
            
            # Additional metrics for "Active Frames" (Frames that had masking)
            err_net_active = 0
            total_sym_active = 0
            
            # For baselines (active frames only)
            err_no_net_active = 0
            err_ideal_active = 0
            
            # For Masked MSE (Focus on masked positions only)
            mse_sum_net_masked = 0
            mse_sum_no_net_masked = 0
            mse_sum_ideal_masked = 0
            total_masked_points = 0

            # Use pre-generated data
            for batch_data in eval_data[snr_db]:
                s_est_masked = batch_data['s_est_masked']
                s_est_ideal = batch_data['s_est_ideal']
                s_orig_tx = batch_data['s_orig_tx']
                mask_idx = batch_data['mask_idx']
                data_test = batch_data['data_test']
                T_conj = batch_data['TRANSFORM_MATRIX'].conj()

                # --- No Net path & Ideal Path ---
                if not baseline_done:
                    # 1. Masked No Net
                    x_est_no_net = torch.matmul(s_est_masked, T_conj)
                    start_idx = 1 + Q
                    end_idx = 1 + Q + D
                    d_est_no_net = x_est_no_net[:, start_idx:end_idx]
                    d_demod_no_net = qam_demod(d_est_no_net, M_mod)
                    batch_err_no_net = (d_demod_no_net != data_test)
                    err_no_net += torch.sum(batch_err_no_net).item()
                    
                    # MSE Calculation for No Net (Baseline)
                    # s_est_masked is the input (no net processing)
                    # We compare it against s_orig_tx ONLY at mask_idx
                    if mask_idx.any():
                        diff_no_net = s_est_masked[mask_idx] - s_orig_tx[mask_idx]
                        mse_sum_no_net_masked += (diff_no_net.real**2 + diff_no_net.imag**2).sum().item()
                    
                    # 2. Ideal No Mask (Upper Bound Performance)
                    x_est_ideal = torch.matmul(s_est_ideal, T_conj)
                    d_est_ideal = x_est_ideal[:, start_idx:end_idx]
                    d_demod_ideal = qam_demod(d_est_ideal, M_mod)
                    batch_err_ideal = (d_demod_ideal != data_test)
                    err_ideal += torch.sum(batch_err_ideal).item()
                    
                    # MSE Calculation for Ideal
                    if mask_idx.any():
                        # Even ideal has noise
                        diff_ideal = s_est_ideal[mask_idx] - s_orig_tx[mask_idx]
                        mse_sum_ideal_masked += (diff_ideal.real**2 + diff_ideal.imag**2).sum().item()

                # --- With Net path ---
                # Full Frame Inference
                with torch.no_grad():
                    reconstructed_frame = model(s_est_masked)
                
                s_est_net = s_est_masked.clone()
                
                # Replace ONLY the masked positions with the model's output
                # This preserves the unmasked (good) symbols
                if mask_idx.any():
                    s_est_net[mask_idx] = reconstructed_frame[mask_idx]
                    
                    # Calculate Masked MSE
                    target_output = s_orig_tx[mask_idx]
                    pred_output = reconstructed_frame[mask_idx]
                    
                    diff = pred_output - target_output
                    sq_diff = (diff.real**2 + diff.imag**2).sum().item()
                    
                    total_mse_sum += sq_diff
                    total_mse_count += mask_idx.sum().item()
                    mse_sum_net_masked += sq_diff
                    total_masked_points += mask_idx.sum().item()
                    
                x_est_net = torch.matmul(s_est_net, T_conj)
                start_idx = 1 + Q
                end_idx = 1 + Q + D
                d_est_net = x_est_net[:, start_idx:end_idx]
                d_demod_net = qam_demod(d_est_net, M_mod)
                
                # Calculate errors
                batch_errors = (d_demod_net != data_test)
                err_net += torch.sum(batch_errors).item()
                total_sym += batch_errors.numel()
                
                # Active Frame Logic:
                # Find which frames in the batch had at least one masked symbol
                # mask_idx shape is (Batch, N)
                frames_with_mask = mask_idx.any(dim=1) # (Batch,) boolean
                if frames_with_mask.any():
                    # Select only rows corresponding to active frames
                    active_errors = batch_errors[frames_with_mask]
                    err_net_active += torch.sum(active_errors).item()
                    total_sym_active += frames_with_mask.sum().item() * D
                    
                    if not baseline_done:
                        active_errors_no_net = batch_err_no_net[frames_with_mask]
                        err_no_net_active += torch.sum(active_errors_no_net).item()
                        
                        active_errors_ideal = batch_err_ideal[frames_with_mask]
                        err_ideal_active += torch.sum(active_errors_ideal).item()

            if total_sym > 0:
                ber = err_net / total_sym
            else:
                ber = 0.0
            ber_list.append(ber)
            
            active_ber_str = "N/A"
            if total_sym_active > 0:
                active_ber = err_net_active / total_sym_active
                active_ber_str = f"{active_ber:.5f}"
                active_ber_list.append(active_ber)
            else:
                active_ber_list.append(0.0)

            # Calculate Average Masked MSE for this SNR
            if total_masked_points > 0:
                avg_masked_mse = mse_sum_net_masked / total_masked_points
            else:
                avg_masked_mse = 0.0
            masked_mse_list.append(avg_masked_mse)

            if not baseline_done:
                if total_sym > 0:
                    ber_no_net_list.append(err_no_net / total_sym)
                    ber_ideal_list.append(err_ideal / total_sym)
                else:
                    ber_no_net_list.append(0.0)
                    ber_ideal_list.append(0.0)
                
                if total_sym_active > 0:
                    active_ber_no_net_list.append(err_no_net_active / total_sym_active)
                    active_ber_ideal_list.append(err_ideal_active / total_sym_active)
                else:
                    active_ber_no_net_list.append(0.0)
                    active_ber_ideal_list.append(0.0)
                
                if total_masked_points > 0:
                    masked_mse_no_net_list.append(mse_sum_no_net_masked / total_masked_points)
                    masked_mse_ideal_list.append(mse_sum_ideal_masked / total_masked_points)
                else:
                    masked_mse_no_net_list.append(0.0)
                    masked_mse_ideal_list.append(0.0)

            print(f"SNR {snr_db}dB: BER={ber:.5f} | Active Frame BER={active_ber_str}")

        # Store metrics
        # Use BER at 20dB (index 4 in 0,5,10,15,20,25) as a representative performance metric
        rep_ber_idx = 4
        if rep_ber_idx < len(ber_list):
            rep_ber = ber_list[rep_ber_idx]
        else:
            rep_ber = ber_list[-1]
            
        # Calculate Average MSE
        avg_mse = total_mse_sum / total_mse_count if total_mse_count > 0 else float('inf')
            
        model_metrics[model_name] = {
            "params": total_params,
            "ber_20db": rep_ber,
            "mse": avg_mse
        }
        
        results[model_name] = ber_list
        results[f"{model_name}_Active"] = active_ber_list
        results[f"{model_name}_MSE"] = masked_mse_list
        
        if not baseline_done:
            results["Masked_NoNet"] = ber_no_net_list
            results["Masked_NoNet_Active"] = active_ber_no_net_list
            results["Masked_NoNet_MSE"] = masked_mse_no_net_list
            
            results["Ideal_NoMask"] = ber_ideal_list
            results["Ideal_NoMask_Active"] = active_ber_ideal_list
            results["Ideal_NoMask_MSE"] = masked_mse_ideal_list
            baseline_done = True
            
    # --- Plot 1: Standard BER ---
    plt.figure(figsize=(10, 8), dpi=100)

    # Define styles for distinction
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Baseline 1: Masked No Net (Lower Bound)
    plt.semilogy(SNR_dB_range, results["Masked_NoNet"], label='Masked (No Net)',
                 linestyle='--', marker='x', color='gray', linewidth=2, markersize=8, alpha=0.7)
                 
    # Baseline 2: Ideal No Mask (Upper Bound)
    plt.semilogy(SNR_dB_range, results["Ideal_NoMask"], label='Ideal (No Mask)',
                 linestyle='-', marker='None', color='black', linewidth=3, alpha=1.0)

    # Models
    for i, (name, ber) in enumerate(models.items()):
        plt.semilogy(SNR_dB_range, results[name], label=name,
                     marker=markers[i % len(markers)], color=colors[i % len(colors)],
                     linewidth=1.5, markersize=7, alpha=0.9)

    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    plt.title('AFDM PAPR Reduction: Neural Network Comparison', fontsize=16, pad=20)
    plt.grid(True, which="major", linestyle='-', alpha=0.5)
    plt.grid(True, which="minor", linestyle=':', alpha=0.3)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.ylim(bottom=1e-6)  # Limit lower bound
    plt.tight_layout()

    save_dir = os.path.join(script_dir, "image")
    os.makedirs(save_dir, exist_ok=True)
    save_gpu_complexity_latency_table(models, device, save_dir)

    # Save per-model wall-clock training time for hardware reporting.
    import csv
    training_time_path = os.path.join(save_dir, "training_time_gpu.csv")
    try:
        with open(training_time_path, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "Model",
                    "Device",
                    "Epochs",
                    "Batches_per_epoch",
                    "Batch_size",
                    "Training_seconds",
                    "Training_minutes",
                ],
            )
            writer.writeheader()
            for row in training_time_records:
                out_row = row.copy()
                out_row["Training_seconds"] = f"{row['Training_seconds']:.2f}"
                out_row["Training_minutes"] = f"{row['Training_minutes']:.4f}"
                writer.writerow(out_row)
        print(f"\n=> Training-time table saved to {training_time_path}")
    except Exception as e:
        print(f"\n=> Failed to save training-time table: {e}")

    save_path = os.path.join(save_dir, "ber_comparison_final.png")
    plt.savefig(save_path, dpi=300)  # High DPI for paper
    print(f"\nFinal publication-quality plot saved to {save_path}")
    
    # --- Plot 2: Active Frames BER Comparison (The "Hard" Plot) ---
    plt.figure(figsize=(10, 8), dpi=100)
    
    # Baseline 1: Masked No Net (Active)
    plt.semilogy(SNR_dB_range, results["Masked_NoNet_Active"], label='Masked (No Net)',
                 linestyle='--', marker='x', color='gray', linewidth=2, markersize=8, alpha=0.7)
                 
    # Baseline 2: Ideal No Mask (Active)
    # Note: Even for active frames, the Ideal curve follows the AWGN trend, but restricted to specific frames
    plt.semilogy(SNR_dB_range, results["Ideal_NoMask_Active"], label='Ideal (No Mask)',
                 linestyle='-', marker='None', color='black', linewidth=3, alpha=1.0)

    # Models (Active)
    for i, model_name in enumerate(models.keys()):
        active_key = f"{model_name}_Active"
        if active_key in results:
            plt.semilogy(SNR_dB_range, results[active_key], label=model_name,
                         marker=markers[i % len(markers)], color=colors[i % len(colors)],
                         linewidth=1.5, markersize=7, alpha=0.9)

    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Active Frame BER (Symbol Error Rate)', fontsize=14, fontweight='bold')
    plt.title('AFDM Neural Network Performance on MASKED Frames Only', fontsize=16, pad=20)
    plt.grid(True, which="major", linestyle='-', alpha=0.5)
    plt.grid(True, which="minor", linestyle=':', alpha=0.3)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.ylim(bottom=1e-6)
    plt.tight_layout()

    save_path_active = os.path.join(save_dir, "ber_comparison_active.png")
    plt.savefig(save_path_active, dpi=300)
    print(f"Active Frame comparison plot saved to {save_path_active}")

    # --- Plot 3: Masked Region MSE Comparison (Hardest Metric) ---
    plt.figure(figsize=(10, 8), dpi=100)
    
    # Baseline 1
    plt.semilogy(SNR_dB_range, results["Masked_NoNet_MSE"], label='Masked (No Net)',
                 linestyle='--', marker='x', color='gray', linewidth=2, markersize=8, alpha=0.7)
    # Baseline 2
    plt.semilogy(SNR_dB_range, results["Ideal_NoMask_MSE"], label='Ideal (No Mask)',
                 linestyle='-', marker='None', color='black', linewidth=3, alpha=1.0)

    # Models
    for i, model_name in enumerate(models.keys()):
        mse_key = f"{model_name}_MSE"
        if mse_key in results:
            plt.semilogy(SNR_dB_range, results[mse_key], label=model_name,
                         marker=markers[i % len(markers)], color=colors[i % len(colors)],
                         linewidth=1.5, markersize=7, alpha=0.9)

    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Masked Region MSE', fontsize=14, fontweight='bold')
    plt.title('Reconstruction Error on MASKED POSITIONS Only', fontsize=16, pad=20)
    plt.grid(True, which="major", linestyle='-', alpha=0.5)
    plt.grid(True, which="minor", linestyle=':', alpha=0.3)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.tight_layout()

    save_path_mse = os.path.join(save_dir, "masked_mse_comparison.png")
    plt.savefig(save_path_mse, dpi=300)
    print(f"Masked MSE comparison plot saved to {save_path_mse}")

    # --- Plot 4: Training Loss Convergence Curve (新增：绘制训练损失收敛曲线) ---
    # Temporarily import matplotlib just for font settings here to avoid global change
    import matplotlib
    original_font_family = matplotlib.rcParams['font.family']
    original_math_fontset = matplotlib.rcParams['mathtext.fontset']
    original_unicode_minus = matplotlib.rcParams['axes.unicode_minus']
    
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 8), dpi=100)
    for i, model_name in enumerate(models.keys()):
        if model_name in loss_history:
            # 取对数刻度可以更清晰地看出收敛趋势
            plt.semilogy(range(1, EPOCHS + 1), loss_history[model_name], label=model_name,
                         color=colors[i % len(colors)], linewidth=2, alpha=0.9)

    plt.xlabel('Training Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Training Loss (MSE on Masked Symbols)', fontsize=14, fontweight='bold')
    plt.title('Convergence Speed Comparison: Training Loss vs Epochs\n(Fixed SNR: 20dB)', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper right', framealpha=0.9)
    
    # 增加边框粗细，增加学术感
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    plt.tight_layout()

    save_path_loss = os.path.join(save_dir, "loss_convergence_fixed.png")
    plt.savefig(save_path_loss, dpi=300)
    print(f"Training Loss convergence plot (Fixed SNR) saved to {save_path_loss}")

    # Restore original matplotlib settings
    matplotlib.rcParams['font.family'] = original_font_family
    matplotlib.rcParams['mathtext.fontset'] = original_math_fontset
    matplotlib.rcParams['axes.unicode_minus'] = original_unicode_minus

    # --- Plot 5: Gradient Norm Before Clipping (新增：绘制截断前梯度范数) ---
    plt.figure(figsize=(10, 8), dpi=100)
    for i, model_name in enumerate(models.keys()):
        if model_name in grad_norm_history:
            # 同样采用对数坐标来容纳可能出现的巨大数值
            plt.semilogy(range(1, EPOCHS + 1), grad_norm_history[model_name], label=model_name,
                         color=colors[i % len(colors)], linewidth=2, alpha=0.8)
                         
    # 画一条红色虚线表示我们设置的截断阈值
    plt.axhline(y=1.0, color='r', linestyle='--', label='Clipping Threshold (1.0)', linewidth=2.5)

    plt.xlabel('Training Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Max Gradient $L_2$ Norm (Peak per Epoch)', fontsize=14, fontweight='bold')
    plt.title('Gradient Explosion Risk: Peak Unclipped Gradient Norms (Fixed 20dB SNR)', fontsize=16, pad=20)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper right', framealpha=0.9)
    plt.tight_layout()

    save_path_grad = os.path.join(save_dir, "grad_norm_history.png")
    plt.savefig(save_path_grad, dpi=300)
    print(f"Gradient norm history plot saved to {save_path_grad}")

    # --- Save BER Results to CSV (Excel Compatible) (新增：导出误码率表格) ---
    import csv
    csv_file_path = os.path.join(save_dir, "ber_results.csv")
    try:
        # 使用 utf-8-sig 编码，确保用 Excel 打开时表头中文不会乱码
        with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            
            # 写入表头
            header = ['SNR (dB)', 'Ideal_NoMask (理想下限)', 'Masked_NoNet (未恢复上限)'] + list(models.keys())
            writer.writerow(header)
            
            # 逐行写入对应 SNR 的数据
            for i, snr in enumerate(SNR_dB_range):
                row = [snr]
                row.append(results["Ideal_NoMask"][i])
                row.append(results["Masked_NoNet"][i])
                for model_name in models.keys():
                    row.append(results[model_name][i])
                writer.writerow(row)
                
        print(f"\n=> 成功：各信噪比点的误码率(BER)已导出至表格 {csv_file_path} (可直接双击用 Excel 打开)")
    except Exception as e:
        print(f"\n=> 导出 BER 表格失败: {e}")

    # --- Print Performance Summary Table ---
    print("\n" + "="*85)
    print(f"{'Model Name':<15} | {'Params':<10} | {'BER @ 20dB':<12} | {'MSE':<12} | {'Score':<20}")
    print("-" * 85)
    
    # Calculate a simple efficiency score: 1 / (BER * Params)
    # (Lower BER and Lower Params -> Higher Score)
    
    for name, metrics in model_metrics.items():
        p = metrics['params']
        b = metrics['ber_20db']
        m = metrics['mse']
        # Avoid division by zero
        if b == 0: b = 1e-9
        
        # Heuristic Score: 1e6 / (BER * Params)
        # You can also incorporate MSE into the score if desired, e.g., 1e6 / (MSE * Params)
        # For now keeping BER-based score but displaying MSE
        score = 1e6 / (b * p)
        
        print(f"{name:<15} | {p:<10} | {b:<12.5f} | {m:<12.6f} | {score:<20.2f}")
    print("="*85)
    print("Score Definition: 1e6 / (BER * Params). Higher is better.")
    print("MSE is Mean Squared Error of reconstructed symbols (Lower is better).")

    # --- Print Full BER Data for All SNRs ---
    print("\n" + "="*100)
    print("FULL BER RESULTS ACROSS ALL SNRs (各信噪比误码率汇总，可直接复制粘贴)")
    print("="*100)
    snr_list_str = ", ".join([f"{snr:>8d}" for snr in SNR_dB_range])
    print(f"{'SNR_dB':<20} = [{snr_list_str}]")
    print("-" * 100)
    
    # Print Baselines
    if "Ideal_NoMask" in results:
        ideal_ber_str = ", ".join([f"{b:>8.6f}" for b in results["Ideal_NoMask"]])
        print(f"{'Ideal_NoMask':<20} = [{ideal_ber_str}]")
        
    if "Masked_NoNet" in results:
        nonet_ber_str = ", ".join([f"{b:>8.6f}" for b in results["Masked_NoNet"]])
        print(f"{'Masked_NoNet':<20} = [{nonet_ber_str}]")
        
    print("-" * 100)
    # Print Models
    for model_name in models.keys():
        if model_name in results:
            model_ber_str = ", ".join([f"{b:>8.6f}" for b in results[model_name]])
            # 替换横杠以防作为Python变量名时不合法
            var_name = model_name.replace('-', '_')
            print(f"{var_name:<20} = [{model_ber_str}]")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
