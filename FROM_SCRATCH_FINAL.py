# ================================================
#                    LIBRARIES                    
# ================================================

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import torch
from tqdm import tqdm
from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import argparse
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from xgboost import XGBRegressor

# =================================================
#                    SEED SETUP
# =================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =================================================
#                  DATA & HELPERS                    
# =================================================

def generate_k2(tau_c, T):
    x = T / tau_c
    return (np.exp(-2 * x) - 1 + 2 * x) / (2 * x ** 2)

def k2_model(T, tau_c_fit):
    x = T / tau_c_fit
    return (np.exp(-2 * x) - 1 + 2 * x) / (2 * x ** 2)

def noise(frames, k2_vals):
    sigma = np.sqrt(k2_vals * (k2_vals + 0.5) / frames)
    noise = sigma * np.random.randn(len(k2_vals))
    return k2_vals + noise

def create_mask(no_points_total, miss_rate):
    mask = np.zeros(no_points_total)
    num_observed = int(no_points_total * (1 - miss_rate))
    indices = random.sample(range(no_points_total), num_observed)
    mask[indices] = 1
    return mask

def data_loader(data, miss_rate):
    no, dim = data.shape
    data_m = np.array([create_mask(dim, miss_rate) for _ in range(no)])
    miss_data = data.copy()
    miss_data[data_m == 0] = np.nan
    return data, miss_data, data_m

def dk2_dtauc(tau_c, T):
    x = T / tau_c
    dx_dtau = -T / (tau_c**2)
    dk2_dx = (2 * (x - 1) * np.exp(-2*x) + 2) / (2 * x**3)
    return dk2_dx * dx_dtau

def cramer_rao_bound(tau_c, T, N_frames):
    k2_vals = generate_k2(tau_c, T)
    sig2 = k2_vals * (k2_vals + 0.5) / N_frames
    dk2_dtau = dk2_dtauc(tau_c, T)
    fisher_info = np.sum((dk2_dtau**2) / sig2)
    return 1 / fisher_info if fisher_info != 0 else np.inf

# =================================================
#       TRAIN RANDOM FOREST TAU_C PREDICTOR        
# =================================================

def train_tau_c_model(num_curves=20000, T=None):
    tau_c_train = np.linspace(0.05, 1.5, num_curves)
    X = []
    for tau in tau_c_train:
        clean_k2 = generate_k2(tau, T)
        noisy_k2 = noise(args.N_frames, clean_k2)
        X.append(noisy_k2)
    X = np.array(X)
    y = tau_c_train

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    model = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Train R^2: {model.score(X_train, y_train):.4f}, Test R^2: {model.score(X_test, y_test):.4f}")
    return model

# =================================================
#                    GAIN MODEL 
# =================================================

def gain(data, gain_parameters, data_m,return_mse=False):
    data_m = data_m.astype(np.float32)
    batch_size = gain_parameters['batch_size']
    miss_rate = gain_parameters['miss_rate']
    hint_rate = gain_parameters['hint_rate']
    iterations = gain_parameters['iterations']
    alpha = gain_parameters['alpha']
    
    no, dim = data.shape

    h_dim = int(dim)

    norm_data = data.copy()
    norm_data_x = np.nan_to_num(norm_data, 0)

    # PLACEHOLDERS
    X = tf.placeholder(tf.float32, shape=[None, dim])
    H = tf.placeholder(tf.float32, shape=[None, dim])
    M = tf.placeholder(tf.float32, shape=[None, dim])

    # DISCRIMINATOR
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    D_b1 = tf.Variable(tf.zeros([h_dim]))
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros([h_dim]))
    D_W3 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b3 = tf.Variable(tf.zeros([h_dim]))
    D_W4 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b4 = tf.Variable(tf.zeros([h_dim]))
    D_W5 = tf.Variable(xavier_init([h_dim, dim]))
    D_b5 = tf.Variable(tf.zeros([dim]))
    theta_D = [D_W1, D_W2, D_W3, D_W4, D_W5, D_b1, D_b2, D_b3, D_b4, D_b5]

    # GENERATOR
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros([h_dim]))
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros([h_dim]))
    G_W3 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b3 = tf.Variable(tf.zeros([h_dim]))
    G_W4 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b4 = tf.Variable(tf.zeros([h_dim]))
    G_W5 = tf.Variable(xavier_init([h_dim, dim]))
    G_b5 = tf.Variable(tf.zeros([dim]))
    theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_b1, G_b2, G_b3, G_b4, G_b5]

    def generator(x, m):
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
        G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
        G_prob = tf.nn.sigmoid(tf.matmul(G_h4, G_W5) + G_b5)
        return G_prob

    def discriminator(x, h):
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
        D_h4 = tf.nn.relu(tf.matmul(D_h3, D_W4) + D_b4)
        D_logit = tf.matmul(D_h4, D_W5) + D_b5
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    G_sample = generator(X, M)
    Hat_X = X * M + G_sample * (1 - M)
    D_prob = discriminator(Hat_X, H)

    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for it in tqdm(range(iterations)):
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]

        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp], feed_dict={M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss], feed_dict={X: X_mb, M: M_mb, H: H_mb})
        # if (it%100==0):
        #     print(f'D_loss   =     {D_loss_curr}    |  G_loss   =    {G_loss_curr}   |  MSE_loss   =    {MSE_loss_curr}')
        # if np.isnan(D_loss_curr) or np.isnan(G_loss_curr) or np.isnan(MSE_loss_curr):
        #     print(f"NaN detected at iteration {it} -- stopping early.")
        #     break

    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
    sess.close()
    return (imputed_data, MSE_loss_curr) if return_mse else imputed_data

# =================================================
#                MAIN FUNCTION
# =================================================

def main(args):
    tau_c_values = np.arange(0.1, 1.5, 0.05)
    # tau_c_values=[0.6,1.4]
    T = np.logspace(-1, 1, 50)
    num_curves = 10000
    miss_rate_list = np.array([0.75,0.7,0.65,0.6,0.55,0.5])

    # === Create folder for saving plots ===
    save_dir = "plots 100 "
    os.makedirs(save_dir, exist_ok=True)

    print("Training ML model to predict tau_c...")
    # model = train_tau_c_model(num_curves=20000, T=T)
    # model1 = train_tau_c_model(num_curves=20000, T=T)
    avg_mse_orig_vs_true = np.zeros(len(miss_rate_list))
    avg_mse_imp_vs_true = np.zeros(len(miss_rate_list))
    avg_tau_c_dev_orig = np.zeros(len(miss_rate_list))
    avg_tau_c_dev_imp = np.zeros(len(miss_rate_list))
    # === Fixed global exposure time (50 points) ===
    T_full = np.logspace(-1, 1, 50)

    # === Train Model 1 (full 50-point curves) once ===
    print("\n[Model 1] Training on full 50-point exposure times (for imputed data)...")
    model_full = train_tau_c_model(num_curves=20000, T=T_full)

    # === Train Model 2 (varying reduced T) for each miss rate ===
    model_dict = {}
    for miss_rate in miss_rate_list:
        num_points = int((1 - miss_rate) * 50)
        T_reduced = np.logspace(-1, 1, num_points)
        print(f"\n[Model 2] Training on {num_points} points (Miss rate = {miss_rate})...")
        model_reduced = train_tau_c_model(num_curves=20000, T=T_reduced)
        model_dict[miss_rate] = (model_reduced, T_reduced)

    crb_list = np.zeros((len(tau_c_values), len(miss_rate_list)))
    bias_orig = np.zeros((len(tau_c_values), len(miss_rate_list)))
    bias_imp = np.zeros((len(tau_c_values), len(miss_rate_list)))

    for t_idx, tau_c_fixed in enumerate(tau_c_values):

        fig, axs = plt.subplots(1, len(miss_rate_list)+1, figsize=(5 * (len(miss_rate_list)+1), 5))
        fig.suptitle(f'k² Curve Comparisons & MSE Plot for τ_c = {tau_c_fixed:.2f}', fontsize=16)

        mse_orig_vs_true_list = []
        mse_imp_vs_true_list = []

        for i, miss_rate in enumerate(miss_rate_list):
            print(f'\n--- Running GAIN + τ_c estimation for τ_c={tau_c_fixed} with miss_rate={miss_rate} ---')

            gain_parameters = {
                'batch_size': args.batch_size,
                'miss_rate': miss_rate,
                'hint_rate': args.hint_rate,
                'alpha': args.alpha,
                'iterations': args.iterations,
                'N_frames': args.N_frames
            }

            T_reduced = model_dict[miss_rate][1]
            model_reduced = model_dict[miss_rate][0]

            # Generate and mask data
            k2_vals_all = np.array([generate_k2(tau_c_fixed, T_full) for _ in range(num_curves)])
            data, miss_k2, data_m = data_loader(k2_vals_all, miss_rate)

            # GAIN imputation
            imputed_data, _ = gain(miss_k2, gain_parameters, data_m, return_mse=True)

            # Add noise
            noisy_original = noise(args.N_frames, miss_k2[0])
            noisy_imputed = noise(args.N_frames, imputed_data[0])

            # τ_c prediction# Select only the observed T-points for model_reduced
            observed_idx = np.where(data_m[0] == 1)[0]
            noisy_original_reduced = noisy_original[observed_idx]
            tau_c_pred_orig = model_reduced.predict(noisy_original_reduced.reshape(1, -1))[0]

            tau_c_pred_imp = model_full.predict(noisy_imputed.reshape(1, -1))[0]
            crb = cramer_rao_bound(tau_c_fixed, T_full, args.N_frames)
            crb_list[t_idx, i] = crb

            bias_orig[t_idx, i] = tau_c_pred_orig - tau_c_fixed
            bias_imp[t_idx, i] = tau_c_pred_imp - tau_c_fixed

            # Compute true k² (ground truth)
            true_k2 = generate_k2(tau_c_fixed, T_full)
            mse_orig_vs_true = np.mean((k2_model(T_full, tau_c_pred_orig) - true_k2) ** 2)
            mse_imp_vs_true = np.mean((k2_model(T_full, tau_c_pred_imp) - true_k2) ** 2)

            # τ_c Deviation
            dev_orig = abs(tau_c_pred_orig - tau_c_fixed) / tau_c_fixed * 100
            dev_imp = abs(tau_c_pred_imp - tau_c_fixed) / tau_c_fixed * 100
            avg_tau_c_dev_orig[i] += dev_orig
            avg_tau_c_dev_imp[i] += dev_imp

            # Plotting
            axs[i].plot(T_full, true_k2, label='True', linewidth=2)
            axs[i].plot(T_full[observed_idx], noisy_original[observed_idx], 'o', label='Noisy Original')
            axs[i].plot(T_full, imputed_data[0], 'x--', label='Imputed')
            axs[i].set_xscale('log')
            axs[i].set_xlabel('Exposure Time T')
            axs[i].set_ylabel('k²')
            axs[i].set_title(f'Miss Rate = {miss_rate:.2f}')
            axs[i].legend()
            axs[i].grid(True)

            mse_orig_vs_true_list.append(mse_orig_vs_true)
            mse_imp_vs_true_list.append(mse_imp_vs_true)
            avg_mse_orig_vs_true[i] += mse_orig_vs_true
            avg_mse_imp_vs_true[i] += mse_imp_vs_true


        axs[-1].plot(miss_rate_list, mse_orig_vs_true_list, 'o-', label='Original MSE')
        axs[-1].plot(miss_rate_list, mse_imp_vs_true_list, 's--', label='Imputed MSE')
        axs[-1].set_xlabel('Miss Rate')
        axs[-1].set_ylabel('MSE vs True k²')
        axs[-1].set_title('MSE Comparison')
        axs[-1].legend()
        axs[-1].grid(True)
        axs[-1].invert_xaxis()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # === Save figure per tau_c ===
        filename = f"tau_c_{tau_c_fixed:.2f}_comparison.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(fig)

    # === FINAL AVERAGE PLOTS ===
    avg_mse_orig_vs_true /= len(tau_c_values)
    avg_mse_imp_vs_true /= len(tau_c_values)
    avg_tau_c_dev_orig /= len(tau_c_values)
    avg_tau_c_dev_imp /= len(tau_c_values)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Average Metrics vs Miss Rate", fontsize=16)

    axs[0].plot(miss_rate_list, avg_mse_orig_vs_true, 'o-', label='Original')
    axs[0].plot(miss_rate_list, avg_mse_imp_vs_true, 's--', label='Imputed')
    axs[0].set_xlabel("Miss Rate")
    axs[0].set_ylabel("Avg MSE vs True k²")
    axs[0].set_title("Average MSE")
    axs[0].invert_xaxis()
    axs[0].legend()
    axs[0].grid(True)

    bar_width = 0.35
    x = np.arange(len(miss_rate_list))
    known_points = [int((1 - mr) * 50) for mr in miss_rate_list]

    axs[1].bar(x - bar_width/2, avg_tau_c_dev_orig, bar_width, label='Original')
    axs[1].bar(x + bar_width/2, avg_tau_c_dev_imp, bar_width, label='Imputed')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([str(kp) for kp in known_points])
    axs[1].set_xlabel("Number of Known (Unmasked) Points")
    axs[1].set_ylabel("Avg Deviation in τ_c (%)")
    axs[1].set_title("Average τ_c Estimation Deviation")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # === Save final average plot ===
    plt.savefig(os.path.join(save_dir, "average_mse_and_tau_c_deviation.png"))
    plt.close(fig)

    # ==== cramer rao bound ============= #
    
        # === CRB and Bias Plot ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Average Bias and CRB vs Miss Rate", fontsize=16)

    axs[0].plot(miss_rate_list, avg_mse_orig_vs_true, 'o-', label='Original MSE')
    axs[0].plot(miss_rate_list, avg_mse_imp_vs_true, 's--', label='Imputed MSE')
    axs[0].plot(miss_rate_list, np.mean((crb_list), axis=0), 'd-', color='purple', label=f'CRB {np.mean(crb_list[0]):.6f}')
    axs[0].set_xlabel("Miss Rate")
    axs[0].set_ylabel("BIAS OR MSE VALUES")
    axs[0].invert_xaxis()
    axs[0].grid(True)
    axs[0].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "crb_and_bias_plot.png"))
    plt.close(fig)


# =================================================
#                EXECUTION ENTRY
# =================================================

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--hint_rate', default=0.9, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--N_frames', default=100, type=int)
    parser.add_argument('--iterations', default=3500, type=int)
    parser.add_argument('--alpha', default=100, type=int)
    args = parser.parse_args()
    main(args)
