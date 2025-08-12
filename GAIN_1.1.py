import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random

# ==========================
# 0. SEED SETUP
# ==========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================
# 1. DATA GENERATION
# ==========================
N = 7
T = np.logspace(np.log10(0.1), np.log10(10.0), N)  # log-spaced T
tau_c_fixed = 0.1

def generate_k2(tau_c, T):
    x = T / tau_c
    return (np.exp(-2 * x) - 1 + 2 * x) / (2 * x ** 2)

k2_vals = generate_k2(tau_c_fixed, T)

# Add Gaussian noise to k2 values
noise_std = 0.005  # standard deviation of noise
k2_vals_noisy = k2_vals + np.random.normal(0, noise_std, size=k2_vals.shape)

# Normalize inputs
scaler_x = MinMaxScaler()
T_scaled = scaler_x.fit_transform(T.reshape(-1, 1))

scaler_y = StandardScaler()
k2_scaled = scaler_y.fit_transform(k2_vals_noisy.reshape(-1, 1))

X_real = torch.tensor(np.hstack((T_scaled, k2_scaled)), dtype=torch.float32)
dataset = TensorDataset(X_real)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ==========================
# 2. MODELS
# ==========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, t):
        return self.model(t)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# ==========================
# 3. LOSS FUNCTIONS
# ==========================
def discriminator_loss(real_scores, fake_scores):
    return fake_scores.mean() - real_scores.mean()

def generator_loss(fake_scores):
    return -fake_scores.mean()

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1)
    alpha = alpha.expand_as(real_samples)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    grad_outputs = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ==========================
# 4. INIT MODELS & OPTIMIZERS
# ==========================
G = Generator()
D = Discriminator()

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

G.apply(weights_init)
D.apply(weights_init)

optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

# ==========================
# 5. TRAINING LOOP
# ==========================
lambda_gp = 10
epochs = 3000

for epoch in range(epochs):
    for batch in dataloader:
        real_data = batch[0]
        T_real = real_data[:, 0:1]
        k2_real = real_data[:, 1:2]

        # Train Discriminator
        optimizer_D.zero_grad()
        fake_k2 = G(T_real).detach()
        D_real = D(real_data)
        D_fake = D(torch.cat((T_real, fake_k2), dim=1))
        gp = compute_gradient_penalty(D, real_data, torch.cat((T_real, fake_k2), dim=1))
        d_loss = discriminator_loss(D_real, D_fake) + lambda_gp * gp
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_k2 = G(T_real)
        g_loss = generator_loss(D(torch.cat((T_real, fake_k2), dim=1)))
        g_loss.backward()
        optimizer_G.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# ==========================
# 6. IMPUTATION
# ==========================
T_interp = []
for i in range(len(T_scaled) - 1):
    start, end = T_scaled[i], T_scaled[i + 1]
    T_interp.extend(np.linspace(start, end, 5)[1:-1])

T_interp = np.array(T_interp).reshape(-1, 1)
T_interp_tensor = torch.tensor(T_interp, dtype=torch.float32)

with torch.no_grad():
    k2_interp_scaled = G(T_interp_tensor).detach().numpy()

k2_interp = scaler_y.inverse_transform(k2_interp_scaled)
T_interp_inv = scaler_x.inverse_transform(T_interp)

# ==========================
# 7. PLOT RESULTS
# ==========================
plt.figure(figsize=(10, 6))

# Plot true noiseless function
T_dense = np.logspace(np.log10(0.1), np.log10(10.0), 1000)
plt.plot(T_dense, generate_k2(tau_c_fixed, T_dense), 'g--', alpha=0.6, label='True κ² Curve')

# Original points (noisy)
plt.plot(T, k2_vals_noisy, 'ko', label='Noisy κ² (training data)', markersize=6, alpha=0.8)

# True (noiseless) κ² values
plt.plot(T, k2_vals, 'bo-', label='True κ² (noiseless)', markersize=5)

# GAN Imputed values
plt.plot(T_interp_inv, k2_interp, 'rx', label='GAN Imputed κ²', markersize=6)

plt.xlabel('T (exposure time)')
plt.ylabel('κ²')
plt.xscale('log')
plt.title(f'WGAN-GP κ² Imputation (τ_c = {tau_c_fixed*1000:.1f} ms)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()
