import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.preprocessing import OrdinalEncoder


##############################################################################################################################


def load_data() -> pd.DataFrame: # load csv file
    return pd.read_csv("Data.csv", keep_default_na=False)
    # keep_default_na=False ensures data with N/A are interpretted as strings not missing numbers.

    
def col_type(df) -> pd.DataFrame: # Seperate features into different types 
    
    cat_cols = ["Cross_section", "Test_Type", "Reinforcement_Design", "Longitudinal_Type", "End_Anchorage", "Stirrup_Type", 
                "Cement_Type", "Corrosion_Method", "Cathode_Type", "Wet_Dry_Ratio"] # Categorical Features
    num_cols = ["Sustained Loading", "W (mm)", "D (mm)", "L (mm)", "Side Cover (mm)", "Bottom Cover (mm)", "Top Cover (mm)", 
                "Tension Ratio (%)", "Comp Ratio (%)", "fy (MPa)", "fsu (MPa)", "Volumetric Ratio", "W/C Ratio", "fc (MPa)", 
                "Lc (mm)", "Icorr", "Duration (days)", "Solution Concentration", "Immersion Depth (mm)", "Mass Loss (%)", 
                "Sample Length (mm)", "Shear Span (mm)"] # Numerical Features
    out_col = ["ln(M_max_exp)"] # Output Feature [target] - transformed moment capacity
    return cat_cols, num_cols, out_col
    
    
def encoder(df): # Encode categorical variables
    enc = OrdinalEncoder()
    df = enc.fit_transform(df)
    return df


def feature_scaling(df): # Scale all features
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled


#### Error Functions ####

def r_squared(Y, y_hat):
    y_bar = Y.mean()
    ss_res = ((Y - y_hat)**2).sum()
    ss_tot = ((Y - y_bar)**2).sum()
    return 1 - (ss_res/ss_tot)

def mean_squared_err(Y, y_hat):
    var = ((Y - y_hat)**2).sum()
    n = len(Y)
    return var/n

def root_mean_squared_err(Y, y_hat):
    MSE = mean_squared_err(Y, y_hat)
    return np.sqrt(MSE)

def mean_abs_err(Y, y_hat):
    abs_var = (np.abs(Y - y_hat)).sum()
    n = len(Y)
    return abs_var/n












