from Functions import *

import pandas as pd
import numpy as np

import tensorflow as tf
import os
import time

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from pygam import GAM, LinearGAM, s, f, te # pyGAM package for running linear regression GAMs

from pyearth import Earth # pyEarth package for running MARS implementation

%matplotlib inline
import matplotlib.pyplot as plt


############################################# DATA PREPARATION ###################################################


df = load_data() # Load database
cat_cols, num_cols, out_col = col_type(df) # Separate features by type - lists


# Encode categorical variables
df[cat_cols] = encoder(df[cat_cols])
df[cat_cols] = df[cat_cols].astype(object) # Convert to object


# Normalize features and create dataframe of all input variables
X = pd.concat([df[num_cols], df[cat_cols]], axis='columns') 
X_s = feature_scaling(X)
Y = df[out_col]
Y = Y.to_numpy()


############################################## MODELS ####################################################
######## CREATE CROSS-VALIDATION LOOP ########
# Split dataset into k-folds (10) and remove the test set (10 %). 
# Then train all models on the same 90 % of the data, splitting this further into training and validation sets for e.g. the ANN. 
# Reiterate the model over every fold, training and testing each model 10 times. 
# Repeat the 10-fold cross-validation 10 times (100-fold total) reshuffling the dataset split each 10 folds.


kf_init = 10 # Initialize the k-fold split

r2, mse, rmse, mae = np.zeros([100,6]), np.zeros([100,6]), np.zeros([100,6]), np.zeros([100,6])

# Empty lists for combining test sets and predictions
# Insert lists in best performing model to plot model regression, residuals, etc.
Y_test_all = []
y_gbrt_all = []

time_start = time.time()

i = 0
fold_shuffle = np.random.randint(10,100,10)

for j in range(kf_init):
    
    kf = KFold(n_splits=10, random_state=fold_shuffle[j], shuffle=True) # Define the fold split and reshuffle each loop.
    
    for train_index, test_index in kf.split(X_s, Y):
        
        X_train, X_test = X_s[train_index], X_s[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_inv = np.exp(Y_test) # Convert test set back into original scale

        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

        Y_test_list = Y_inv.tolist()
        Y_test_all += Y_test_list

        ##################################################### ANN ##########################################################

        def ann_opt(X_train, X_val, Y_train, Y_val):

            def ann_architecture(): # Create ANN model architecture

                ann_model = Sequential()
                ann_model.add(Dense(256, input_dim=32, kernel_initializer='normal', activation='relu')) 
                ann_model.add(Dense(128, kernel_initializer='normal', activation='relu'))
                ann_model.add(Dense(1))
                ann_model.compile(optimizer='nadam', loss='mse', metrics=['mse', 'mae'])  

                return ann_model

            ann_model = ann_architecture()

            # Train and evaluate the model
            history = ann_model.fit(X_train, Y_train, batch_size=25, epochs=100, verbose=0, validation_data=(X_val, Y_val))
            ann_model.evaluate(X_test, Y_test)

            # Generate predictions
            y_pred_ann = ann_model.predict(X_test)

            return(y_pred_ann, history, ann_model)

        y_pred_ann, history, ann_model = ann_opt(X_train, X_val, Y_train, Y_val)

        # Convert prediction back to original magnitude
        y_ann_inv = np.exp(y_pred_ann)

        # Record error metrics from each fold
        r2[i,0] = r_squared(Y_inv, y_ann_inv)
        mse[i,0] = mean_squared_err(Y_inv, y_ann_inv)
        rmse[i,0] = root_mean_squared_err(Y_inv, y_ann_inv)
        mae[i,0] = mean_abs_err(Y_inv, y_ann_inv)

        ################################################# GBRT ########################################################### 

        def gbrt_opt(X_train, Y_train):

            def gbrt_model(): # Add optimized hyperparameters into GBRT model

                gbrt_model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.2, max_depth=2, 
                                                       max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=4,
                                                       random_state=0, loss='squared_error')            
                return gbrt_model

            gbrt_model = gbrt_model()    

            # Train the optimised model
            gbrt_model.fit(X_train, Y_train.ravel())

            # Predict the response 
            y_pred_gbrt = gbrt_model.predict(X_test)

            return y_pred_gbrt

        y_pred_gbrt = gbrt_opt(X_train, Y_train)

        # Convert prediction back to original magnitude
        y_gbrt_inv = np.exp(y_pred_gbrt).reshape(-1,1)

        # After running 100 iterations, GBRT is the best performing model.
        # Record all predictions into an array for plotting
        y_pred_gbrt_list = y_gbrt_inv.tolist()
        y_gbrt_all += y_pred_gbrt_list

        # Record error metrics from each fold
        r2[i,1] = r_squared(Y_inv, y_gbrt_inv)
        mse[i,1] = mean_squared_err(Y_inv, y_gbrt_inv)
        rmse[i,1] = root_mean_squared_err(Y_inv, y_gbrt_inv)
        mae[i,1] = mean_abs_err(Y_inv, y_gbrt_inv)

        ################################################## RF ############################################################

        def rf_model():

            rf_model = RandomForestRegressor(n_estimators=500, max_depth=7, max_features='auto', min_samples_leaf=1, 
                                             min_samples_split=2, random_state=25, n_jobs=-1, criterion='squared_error')
            return rf_model

        rf_model = rf_model()

        # Train the model
        rf_model.fit(X_train, Y_train.ravel())

        # Predict response
        y_pred_rf = rf_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_rf_inv = np.exp(y_pred_rf).reshape(-1,1)

        # Record error metrics from each fold
        r2[i,2] = r_squared(Y_inv, y_rf_inv)
        mse[i,2] = mean_squared_err(Y_inv, y_rf_inv)
        rmse[i,2] = root_mean_squared_err(Y_inv, y_rf_inv)
        mae[i,2] = mean_abs_err(Y_inv, y_rf_inv)

        #################################################### GAM ##########################################################

        def gam(X_train, Y_train):

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

        gam_model = gam(X_train, Y_train)

        # Train the model
        gam_model.fit(X_train, Y_train)

        # Predict the response
        y_pred_gam = gam_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_gam_inv = np.exp(y_pred_gam).reshape(-1,1)

        # Record error metrics from each fold
        r2[i,3] = r_squared(Y_inv, y_gam_inv)
        mse[i,3] = mean_squared_err(Y_inv, y_gam_inv)
        rmse[i,3] = root_mean_squared_err(Y_inv, y_gam_inv)
        mae[i,3] = mean_abs_err(Y_inv, y_gam_inv)

        ##################################################### MARS ########################################################

        def mars_opt(X_train, Y_train):

            def mars_model():

                mars_model = Earth() # default applies the forward pass, pruning pass, and linear fit

                return mars_model

            mars_model = mars_model()

            # Train the model
            mars_model.fit(X_train, Y_train) 

            # Define grid for hyperparameter tuning
            params_mars = [{'max_degree': [1,2,3]}]

            # Refit model using grid searched hyperparameters
            gs_mars_model = GridSearchCV(Earth(), params_mars, n_jobs=-1).fit(X_train, Y_train)

            return gs_mars_model

        mars_opt = mars_opt(X_train, Y_train)

        # Train the optimized model
        mars_opt.fit(X_train, Y_train)

        # Predict the response
        y_pred_mars = mars_opt.predict(X_test)

        # Convert prediction back to original magnitude
        y_mars_inv = np.exp(y_pred_mars).reshape(-1,1)

        # Record error metrics from each fold
        r2[i,4] = r_squared(Y_inv, y_mars_inv)
        mse[i,4] = mean_squared_err(Y_inv, y_mars_inv)
        rmse[i,4] = root_mean_squared_err(Y_inv, y_mars_inv)
        mae[i,4] = mean_abs_err(Y_inv, y_mars_inv)

        ####################################################### MLR #########################################################

        def mlr():

            mlr_model = LinearRegression()

            return mlr_model

        mlr_model = mlr()

        # Fit the model
        mlr_model.fit(X_train, Y_train)

        # Predict the response
        y_pred_mlr = mlr_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_mlr_inv = np.exp(y_pred_mlr)

        # Record error metrics from each fold
        r2[i,5] = r_squared(Y_inv, y_mlr_inv)
        mse[i,5] = mean_squared_err(Y_inv, y_mlr_inv)
        rmse[i,5] = root_mean_squared_err(Y_inv, y_mlr_inv)
        mae[i,5] = mean_abs_err(Y_inv, y_mlr_inv)
  

        i += 1 
    j += 1
    
    
time_end = time.time()
print("Elapsed time: %.2f seconds" % (time_end - time_start)) 


############################################## MODEL ERROR AND PERFORMANCE ################################################


df_r2 = pd.DataFrame(r2).rename(columns={0:'ANN', 1:'GBRT', 2:'RF', 
                                         3:'GAM', 4:'MARS', 5:'MLR'})
df_mse = pd.DataFrame(mse).rename(columns={0:'ANN', 1:'GBRT', 2:'RF', 
                                         3:'GAM', 4:'MARS', 5:'MLR'})
df_rmse = pd.DataFrame(rmse).rename(columns={0:'ANN', 1:'GBRT', 2:'RF', 
                                         3:'GAM', 4:'MARS', 5:'MLR'})
df_mae = pd.DataFrame(mae).rename(columns={0:'ANN', 1:'GBRT', 2:'RF', 
                                         3:'GAM', 4:'MARS', 5:'MLR'})

# df_r2.to_csv('100-fold R2 - All Models.csv')
# df_mse.to_csv('100-fold MSE - All Models.csv')
# df_rmse.to_csv('100-fold RMSE - All Models.csv')
# df_mae.to_csv('100-fold MAE - All Models.csv')

print(df_r2['GBRT'].mean())
df_r2.head(20)


################################################### MODEL PLOTTING ######################################################
############################################### PREDICITON REGRESSION ###################################################


Y_test_all = np.array(Y_test_all).flatten() # Y_test for full database (10x10 folds)
y_gbrt_all = np.array(y_gbrt_all).flatten() # y_pred for full database (10x10 folds)

# Generate regression lines
reg_line = np.linspace(0, np.max(Y_test_all), len(Y_test_all))
a, b = np.polyfit(Y_test_all, y_gbrt_all, 1)

# Plot regression
fig, ax = plt.subplots(figsize=(8,8))
ax.patch.set_edgecolor('black') # Set border 
ax.patch.set_linewidth(1) # Set border width

# Plot regression lines
ax.plot(reg_line, a*reg_line+b, c='black', linewidth=0.75, linestyle='dashed') # add line of best fit to plot
ax.plot(reg_line, (a*reg_line+b)*0.85, c='red', linewidth=0.75, linestyle='dashed') # add lower bound
ax.plot(reg_line, (a*reg_line+b)*1.15, c='red', linewidth=0.75, linestyle='dashed') # add upper bound
ax.scatter(Y_test_all, y_gbrt_all, c='dodgerblue', marker='o', s=10, alpha=0.9, linewidth=0.1)

leg = ax.legend(["$R^{2}$ = %.3f" %df_r2['GBRT'].mean(), "+/- 15 %"], handlelength=1.2, 
                fontsize=12, loc='upper left')

ax.set_xlabel('$M_{max, exp}$ (kNm)', fontsize=14)
ax.set_ylabel(('$M_{max, pred}$ (kNm)'), fontsize=14)
ax.set_axisbelow(True)
ax.set_xlim(0,150)
ax.set_ylim(0,150)
ax.set_xticks([0,25,50,75,100,125,150], minor=False) # Set x axis increments
ax.set_yticks([0,25,50,75,100,125,150], minor=False) # Set y axis increments
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_axisbelow(True)
ax.grid(which='major', axis='both', linewidth=0.5)

plt.savefig("GBRT Regression", dpi=1200, bbox_inches='tight')
plt.show()


################################################### RESIDUAL ERRORS ######################################################

# Calculate residual error
resid_err = pd.DataFrame(Y_test_all - y_gbrt_all)
mean = np.mean(resid_err)
std = np.std(resid_err)

# Create origin line arrays
origin_x = np.linspace(0, np.max(Y_test_all), len(Y_test_all))
origin_y = np.linspace(0, 0, len(Y_test_all))


fig, ax = plt.subplots(figsize=(8,8))
ax.patch.set_edgecolor('black') # Set border 
ax.patch.set_linewidth(1) # Set border width

ax.plot(origin_x, origin_y, color='black', linewidth=0.75, linestyle='--')
ax.scatter(y_gbrt_all, resid_err, label='Mean, Std', color='red', edgecolor='black', 
             marker='o', s=10, alpha=0.9, linewidth=0.1)

leg = ax.legend(['Mean = %.3f (kNm)' %mean, 'Std = %.3f (kNm)' %std], 
                  fontsize=12, handlelength=0, loc='upper left')
for item in leg.legend_handles:
    item.set_visible(False)

ax.set_xlabel('$M_{max, pred}$ (kNm)', fontsize=14)
ax.set_ylabel(r'$\epsilon_{residual}$ (kNm)', fontsize=14)
ax.set_xlim(0,150)
ax.set_ylim(-30,50)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_axisbelow(True)
ax.grid(which='major', axis='both', linewidth=0.5)

plt.savefig("GBRT Residual Errors", dpi=1200, bbox_inches='tight')
plt.show()


######################################### HYPERPARAMETER OPTIMIZATION EXAMPLES ###########################################
################################################## ANN OPTIMIZATION #####################################################


def ann_architecture(optimizer, activation): # Create ANN model architecture
    ann_model = Sequential()
    ann_model.add(Dense(units = 256, input_dim=32, kernel_initializer='normal', activation='relu')) 
    ann_model.add(Dense(units = 128, kernel_initializer='normal', activation=activation))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])    
    return ann_model

ann_model = KerasRegressor(build_fn = ann_architecture)

# GridSearchCV Optimization
params_ann = {'batch_size': [25,50,75], 
              'epochs': [75,100,125,150,175], 
              'optimizer': ['adam', 'nadam'],
              'activation': ['relu', 'sigmoid']}

# Fit the model to the hyperparameter grid-search
ann_opt = GridSearchCV(estimator = ann_model, param_grid=params_ann, 
                               scoring = 'neg_mean_absolute_error', cv=5, n_jobs=-1)

ann_opt = ann_opt.fit(X_train, Y_train, verbose=0)

print(" Results from ANN Grid Search " )
print("\n The best score across ALL searched params:\n", ann_opt.best_score_)
print("\n The best parameters across ALL searched params:\n", ann_opt.best_params_)


################################################## GBRT OPTIMIZATION #####################################################


def gbrt_bench(): # Create benchmark GBRT model
    gbrt_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=2, max_leaf_nodes=5,
                                           min_samples_leaf=1, min_samples_split=2, random_state=0, loss='squared_error')
    return gbrt_model

gbrt_bench = gbrt_bench()

# Train the benchmark model
gbrt_bench.fit(X_train, Y_train.ravel())

# Apply grid optimisation for other hyperparamters here
params_gbrt = {'n_estimators': [100,200,300,400,500], 
               'learning_rate': [0.01,0.05,0.1,0.15,0.2,0.25], 
               'max_depth': [2,4,6,8], 
               'min_samples_split': [2,4,6,8,10]}

gbrt_opt = GridSearchCV(estimator=gbrt_bench, param_grid=params_gbrt, cv=5, n_jobs=-1)

gbrt_opt.fit(X_train, Y_train)

print(" Results from GBRT Grid Search " )
print("\n The best estimator across ALL searched params:\n", gbrt_opt.best_estimator_)
print("\n The best score across ALL searched params:\n", gbrt_opt.best_score_)
print("\n The best parameters across ALL searched params:\n", gbrt_opt.best_params_)


################################################### RF OPTIMIZATION #####################################################


def rf_bench():
    rf_model = RandomForestRegressor(n_estimators=500, max_depth=5, max_features=1.0, min_samples_leaf=1, 
                                     min_samples_split=2, random_state=25, n_jobs=-1, criterion='squared_error')
    return rf_model

rf_model = rf_bench()

# Define grid search parameters
params_rf = {'n_estimators': [100,200,300,400,500], 
             'max_features': [1.0, 'sqrt', 'log2'], 
             'max_depth': [3,4,5,6,7]}

# Fit the model to the hyperparameter grid-search
rf_opt = GridSearchCV(estimator=rf_model, param_grid=params_rf, cv=5, n_jobs=-1)
rf_opt.fit(X_train, Y_train)

print(" Results from RF Grid Search " )
print("\n The best estimator across ALL searched params:\n", rf_opt.best_estimator_)
print("\n The best score across ALL searched params:\n", rf_opt.best_score_)
print("\n The best parameters across ALL searched params:\n", rf_opt.best_params_)


