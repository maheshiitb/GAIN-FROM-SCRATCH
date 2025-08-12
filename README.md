# GAIN-FROM-SCRATCH
# GAIN-Based τ_c Estimation

This repository contains code for imputing speckle contrast curve values using the **GAIN (Generative Adversarial Imputation Network)** model and estimating τ_c values from the imputed data.

## Files

### 1. `estimation_gain_2.py`
- Uses the GAIN model to impute missing curve values.
- Applies **XGBoost + StandardScaler** to estimate τ_c values from the imputed curves.

### 2. `GAIN_1.1.py`
- Demonstrates how the GAIN model imputes missing values for a **fixed τ_c** value.
- For different τ_c values and different missing rates, it saves the resulting plots as image files.

## 3. `FROM_SCRATCH_FINAL.py`
1. **Data Imputation**  
   - The GAIN model is trained to fill in missing points in the speckle contrast curve.
2. **τ_c Estimation**  
   - The imputed curves are passed through an **XGBoost regression model** (with preprocessing via StandardScaler) to predict τ_c values.
3. **Plot Generation**  
   - For varying τ_c and missing rate combinations, plots of the imputation results are saved as `.png` images.

## Requirements
- Python 3.x
- TensorFlow (for GAIN model)
- XGBoost
- scikit-learn
- NumPy, Pandas, Matplotlib

# Run the GAIN-based τ_c estimation
python estimation_gain_2.py

# Run the GAIN imputation demo for fixed τ_c values
python GAIN_1.1.py
