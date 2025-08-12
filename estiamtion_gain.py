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
# def create_mask(no_points_total, miss_rate):
#     mask = np.zeros(no_points_total)
#     num_observed = int(no_points_total * (1 - miss_rate))
#     if num_observed == 0:
#         return mask
#     step = no_points_total / num_observed
#     indices = [int(i * step) for i in range(num_observed)]
#     indices = sorted(set(min(idx, no_points_total - 1) for idx in indices))
#     mask[indices] = 1
#     return mask

def data_loader(data, miss_rate):
    no, dim = data.shape
    data_m = np.array([create_mask(dim, miss_rate) for _ in range(no)])
    miss_data = data.copy()
    miss_data[data_m == 0] = np.nan
    return data, miss_data, data_m

# =================================================
#       TRAIN RANDOM FOREST TAU_C PREDICTOR        
# =================================================

def train_tau_c_model(num_curves=20000, T=None):
    tau_c_train = np.random.uniform(0.05, 5, num_curves)
    X = []
    for tau in tau_c_train:
        clean_k2 = generate_k2(tau, T)
        noisy_k2 = noise(100, clean_k2)
        X.append(noisy_k2)
    X = np.array(X)
    y = tau_c_train

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    print(f"Train R^2: {model.score(X_train, y_train):.4f}, Test R^2: {model.score(X_test, y_test):.4f}")
    return model

# =================================================
#                    GAIN MODEL 
# =================================================

def gain(data, gain_parameters, data_m, return_mse=False):
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

    X = tf.placeholder(tf.float32, shape=[None, dim])
    H = tf.placeholder(tf.float32, shape=[None, dim])
    M = tf.placeholder(tf.float32, shape=[None, dim])

    def build_mlp(inputs, weights, biases):
        layer = inputs
        for i in range(len(weights) - 1):
            layer = tf.nn.relu(tf.matmul(layer, weights[i]) + biases[i])
        return tf.nn.sigmoid(tf.matmul(layer, weights[-1]) + biases[-1])

    def make_weights_biases():
        weights = [tf.Variable(xavier_init([dim * 2, h_dim]))]
        biases = [tf.Variable(tf.zeros([h_dim]))]
        for _ in range(3):
            weights.append(tf.Variable(xavier_init([h_dim, h_dim])))
            biases.append(tf.Variable(tf.zeros([h_dim])))
        weights.append(tf.Variable(xavier_init([h_dim, dim])))
        biases.append(tf.Variable(tf.zeros([dim])))
        return weights, biases

    G_weights, G_biases = make_weights_biases()
    D_weights, D_biases = make_weights_biases()

    G_sample = build_mlp(tf.concat([X, M], axis=1), G_weights, G_biases)
    Hat_X = X * M + G_sample * (1 - M)
    D_prob = build_mlp(tf.concat([Hat_X, H], axis=1), D_weights, D_biases)

    D_loss = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)
    G_loss = G_loss_temp + alpha * MSE_loss

    D_solver = tf.train.AdamOptimizer(1e-4).minimize(D_loss, var_list=D_weights + D_biases)
    G_solver = tf.train.AdamOptimizer(1e-4).minimize(G_loss, var_list=G_weights + G_biases)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for it in tqdm(range(iterations)):
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]

        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        H_mb = M_mb * binary_sampler(hint_rate, batch_size, dim)
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss], feed_dict={X: X_mb, M: M_mb, H: H_mb})

        if np.isnan(D_loss_curr) or np.isnan(G_loss_curr) or np.isnan(MSE_loss_curr):
            print(f"NaN detected at iteration {it} -- stopping early.")
            break

    imputed_data = sess.run(G_sample, feed_dict={X: norm_data_x, M: data_m})
    sess.close()
    return (imputed_data, MSE_loss_curr) if return_mse else imputed_data

# =================================================
#                MAIN FUNCTION
# =================================================

def main(args):
    # tau_c_values=np.arange(0.1,1.5,0.1)
    tau_c_values = [1.5]
    T = np.logspace(-1, 1, 50)
    num_curves = 10000
    miss_rate_list = np.array([0.9, 0.8, 0.7,0.6])

    print("Training ML model to predict tau_c...")
    model = train_tau_c_model(num_curves=20000, T=T)

    for tau_c_fixed in tau_c_values:
        mse_orig_vs_true_list = []
        mse_imp_vs_true_list = []
        for miss_rate in miss_rate_list:
            gain_parameters = {
                'batch_size': args.batch_size,
                'miss_rate': miss_rate,
                'hint_rate': args.hint_rate,
                'alpha': args.alpha,
                'iterations': args.iterations
            }

            print(f"\nτ_c={tau_c_fixed} | miss_rate={miss_rate}")
            k2_vals_all = np.array([generate_k2(tau_c_fixed, T) for _ in range(num_curves)])
            data, miss_k2, data_m = data_loader(k2_vals_all, miss_rate)

            imputed_data, _ = gain(miss_k2, gain_parameters, data_m, return_mse=True)
            noisy_original = noise(100, miss_k2[0])
            noisy_imputed = noise(100, imputed_data[0])

            # Predict tau_c from model
            tau_c_pred_orig = model.predict(noisy_original.reshape(1, -1))[0]
            tau_c_pred_imp = model.predict(noisy_imputed.reshape(1, -1))[0]

            true_k2 = generate_k2(tau_c_fixed, T)
            mse_orig_vs_true = np.mean((k2_model(T, tau_c_pred_orig) - true_k2) ** 2)
            mse_imp_vs_true = np.mean((k2_model(T, tau_c_pred_imp) - true_k2) ** 2)

            print(f"Predicted τ_c: Original = {tau_c_pred_orig:.4f}, Imputed = {tau_c_pred_imp:.4f}")
            print(f"MSE vs True:  Original = {mse_orig_vs_true:.6f}, Imputed = {mse_imp_vs_true:.6f}")

            mse_orig_vs_true_list.append(mse_orig_vs_true)
            mse_imp_vs_true_list.append(mse_imp_vs_true)

            # Plot curves
            plt.figure(figsize=(10, 6))
            plt.plot(T, data[0] ,label='True')
            plt.plot(T, noisy_original, 'o',label='Noisy Original')
            plt.plot(T, imputed_data[0], 'x--',label='Imputed')
            plt.xscale('log')
            plt.xlabel('Exposure Time T')
            plt.ylabel('k^2')
            plt.title(f'k² Curves | τ_c={tau_c_fixed}, Miss Rate={miss_rate}')
            plt.grid(True)

        # Plot MSE vs Miss Rate
        plt.figure(figsize=(10, 6))
        plt.plot(miss_rate_list, mse_orig_vs_true_list, 'o-', label='Original')
        plt.plot(miss_rate_list, mse_imp_vs_true_list, 's--', label='Imputed')
        plt.xlabel('Miss Rate')
        plt.ylabel('MSE vs True k²')
        plt.title(f'MSE vs Miss Rate | τ_c={tau_c_fixed}')
        plt.grid(True)
        plt.legend()
        plt.gca().invert_xaxis()
        plt.show()

# =================================================
#                EXECUTION ENTRY
# =================================================

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--miss_rate', default=0.6, type=float)
    parser.add_argument('--hint_rate', default=0.99, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--iterations', default=10000, type=int)
    parser.add_argument('--alpha', default=100, type=int)
    args = parser.parse_args()
    main(args)