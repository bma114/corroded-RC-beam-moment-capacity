# Import packages
import pandas as pd
import numpy as np

# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from pygam import LinearGAM, s, f # pyGAM package for linear regression GAMs
# from pyearth import Earth # pyEarth package for running MARS implementation

##############################################################################################################################


# Load the data from csv file
def load_data() -> pd.DataFrame:

    return pd.read_csv("Data.csv", keep_default_na=False)



# Seperate features into different types 
def col_type(df) -> pd.DataFrame: 
    
    # Categorical Features
    cat_cols = ["Cross_section", "Test_Type", "Reinforcement_Design", "Longitudinal_Type", 
                "End_Anchorage", "Stirrup_Type", "Cement_Type", "Corrosion_Method", 
                "Cathode_Type", "Wet_Dry_Ratio"]
    
    # Numerical Features
    num_cols = ["Sustained Loading", "W (mm)", "D (mm)", "L (mm)", "Side Cover (mm)", 
                "Bottom Cover (mm)", "Top Cover (mm)", "Tension Ratio (%)", "Comp Ratio (%)", 
                "fy (MPa)", "fsu (MPa)", "Volumetric Ratio", "W/C Ratio", "fc (MPa)", 
                "Lc (mm)", "Icorr", "Duration (days)", "Solution Concentration", 
                "Immersion Depth (mm)", "Mass Loss (%)", "Sample Length (mm)",
                "Shear Span (mm)"] 
    
    # Output Feature [target] - transformed moment capacity
    out_col = ["ln(M_max_exp)"]

    return cat_cols, num_cols, out_col
    
    

# Encode categorical variables
def encoder(df):

    enc = OrdinalEncoder()
    df = enc.fit_transform(df)

    return df



# Scale all features
def feature_scaling(df):

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    return df_scaled



# Define error metrics
# R^2
def r_squared(Y, y_hat): 

    y_bar = Y.mean()
    ss_res = ((Y - y_hat)**2).sum()
    ss_tot = ((Y - y_bar)**2).sum()

    return 1 - (ss_res/ss_tot)

# MSE
def mean_squared_err(Y, y_hat):

    var = ((Y - y_hat)**2).sum()
    n = len(Y)

    return var/n

# RMSE
def root_mean_squared_err(Y, y_hat):

    MSE = mean_squared_err(Y, y_hat)

    return np.sqrt(MSE)


# MAE
def mean_abs_err(Y, y_hat):

    abs_var = (np.abs(Y - y_hat)).sum()
    n = len(Y)

    return abs_var/n



# Build Artificial Neural Network (ANN) Architecture
def ann_architecture():

    ann_model = Sequential()
    ann_model.add(Dense(256, input_dim=32, kernel_initializer='normal', activation='relu')) 
    ann_model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer='nadam', loss='mse', metrics=['mse', 'mae'])  

    return ann_model



# Build Gradient Boosting Regression Tree (GBRT) model
def build_gbrt():

    gbrt_model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.2, max_depth=2, 
                                            max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=4,
                                            random_state=0, loss='squared_error')            
    return gbrt_model



# Build Random Forest (RF) model
def build_rf():

    rf_model = RandomForestRegressor(n_estimators=500, max_depth=7, max_features='sqrt', min_samples_leaf=1, 
                                        min_samples_split=2, random_state=25, n_jobs=-1, criterion='squared_error')
    return rf_model



# Build Generalized Additive (GAM) model
def build_gam(X_train, Y_train):

    # Define gridsearch parameter ranges
    grid_splines = np.linspace(10,30,20) # number of splines per feature
    lams = np.random.rand(50, 32) # lambda value for smoothing penalization
    lams = lams * 32 - 3 # Search space for lam needs 32 dimensions for a model with 32 lam terms (one per feature)
    lams = np.exp(lams)

    # Build the model
    # Numerical functions given spline terms s(),
    # Categorical variables given step function terms f().
    gam_model = LinearGAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+s(7)+s(8)+s(9)+s(10)+s(11)+s(12)+s(13)+
                            s(14)+s(15)+s(16)+s(17)+s(18)+s(19)+s(20)+s(21)+
                            f(22)+f(23)+f(24)+f(25)+f(26)+f(27)+f(28)+f(29)+f(30)
                            +f(31)).gridsearch(X_train, Y_train, n_splines=grid_splines, lam=lams)

    return gam_model



# Build Multivariate Adaptive Regression Spline (MARS) model
def build_mars():

    mars_model = Earth() # default applies the forward pass, pruning pass, and linear fit

    return mars_model



# Build multiple linear regression (MLR) model
def mlr():

    mlr_model = LinearRegression()

    return mlr_model