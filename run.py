from Functions import *

import time
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

### ========================= ### DATA PREPARATION ### ========================= ###

# Load database
df = load_data()
cat_cols, num_cols, out_col = col_type(df) # Separate features by type - lists


# Encode categorical variables
df[cat_cols] = encoder(df[cat_cols])
df[cat_cols] = df[cat_cols].astype(object) # Convert to object


# Normalize features and create dataframe of all input variables
X = pd.concat([df[num_cols], df[cat_cols]], axis='columns') 
X_s = feature_scaling(X)
Y = df[out_col]
Y = Y.to_numpy()


### ========================= ### TRAIN & TEST MODELS ### ========================= ###
"""

Train, test and evaluate each model based on the full dataset and using a Repeated K-Fold cross-validation approach. 

Repeat the 10-fold cross-validation 10 times (100-fold total) reshuffling the dataset split each 10 folds.

The model performance metrics over all 100-folds and all models are exported as csv files at the end of the script.

Model optimization is excluded form this example to improve computational speed. However, code is provided 
at the end of the script for example implementation of optimization for several key model types. 
Optimization code is commented out by default. Uncomment the code to run the algorithms. 


"""

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
    
    # Define the fold split and reshuffle each loop.
    kf = KFold(n_splits=10, random_state=fold_shuffle[j], shuffle=True) 
    
    for train_index, test_index in kf.split(X_s, Y):
        
        # All features are numerical and normalized
        X_train, X_test = X_s[train_index], X_s[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Convert test set back into original scale
        Y_inv = np.exp(Y_test)

        # Training & Validation datasets for ANN
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

        # Compile test data from all k-folds to plot best model
        Y_test_all.append(Y_inv)



        ### =============================== ### ANN ### =============================== ###

        # Call ANN model
        ann_model = ann_architecture()

        # Train and evaluate the model
        history = ann_model.fit(X_train, Y_train, batch_size=25, epochs=100, verbose=0, validation_data=(X_val, Y_val))
        ann_model.evaluate(X_test, Y_test)

        # Generate predictions
        y_pred_ann = ann_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_ann_inv = np.exp(y_pred_ann)

        # Record error metrics from each fold
        r2[i,0] = r_squared(Y_inv, y_ann_inv)
        mse[i,0] = mean_squared_err(Y_inv, y_ann_inv)
        rmse[i,0] = root_mean_squared_err(Y_inv, y_ann_inv)
        mae[i,0] = mean_abs_err(Y_inv, y_ann_inv)



        ### ============================ ### GBRT ### ============================= ### 

        # Call GBRT model
        gbrt_model = build_gbrt()    

        # Train the optimised model
        gbrt_model.fit(X_train, Y_train.ravel())

        # Predict the response 
        y_pred_gbrt = gbrt_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_gbrt_inv = np.exp(y_pred_gbrt).reshape(-1,1)

        # After running 100 iterations, GBRT is the best performing model.
        # Record all predictions into an array for plotting
        y_gbrt_all.append(y_gbrt_inv)

        # Record error metrics from each fold
        r2[i,1] = r_squared(Y_inv, y_gbrt_inv)
        mse[i,1] = mean_squared_err(Y_inv, y_gbrt_inv)
        rmse[i,1] = root_mean_squared_err(Y_inv, y_gbrt_inv)
        mae[i,1] = mean_abs_err(Y_inv, y_gbrt_inv)



        ### ========================= ### RF ### ========================= ###

        # Call RF model
        rf_model = build_rf()

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



        ### =============================== ### GAM ### =============================== ###

        # Call GAM model
        gam_model = build_gam(X_train, Y_train)

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



        ### =============================== ### MARS ### =============================== ###

        # Call MARS model
        mars_model = build_mars()

        # Train the model
        mars_model.fit(X_train, Y_train) 

        # Define grid for hyperparameter tuning
        params_mars = [{'max_degree': [1,2,3]}]

        # Refit model using grid searched hyperparameters
        gs_mars_model = GridSearchCV(Earth(), params_mars, n_jobs=-1).fit(X_train, Y_train)

        # Train the optimized model
        gs_mars_model.fit(X_train, Y_train)

        # Predict the response
        y_pred_mars = gs_mars_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_mars_inv = np.exp(y_pred_mars).reshape(-1,1)

        # Record error metrics from each fold
        r2[i,4] = r_squared(Y_inv, y_mars_inv)
        mse[i,4] = mean_squared_err(Y_inv, y_mars_inv)
        rmse[i,4] = root_mean_squared_err(Y_inv, y_mars_inv)
        mae[i,4] = mean_abs_err(Y_inv, y_mars_inv)



        ### =============================== ### MLR ### =============================== ###

        # Call MLR model
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
print("Elapsed time: {} minutes and {:.0f} seconds".format
      (int((time_end - time_start) // 60), (time_end - time_start) % 60)) 



## ========================= ### MODEL ERROR AND PERFORMANCE ### ========================= ###


df_r2 = pd.DataFrame(r2).rename(columns={0:'ANN', 1:'GBRT', 2:'RF', 
                                         3:'GAM', 4:'MARS', 5:'MLR'})
df_mse = pd.DataFrame(mse).rename(columns={0:'ANN', 1:'GBRT', 2:'RF', 
                                         3:'GAM', 4:'MARS', 5:'MLR'})
df_rmse = pd.DataFrame(rmse).rename(columns={0:'ANN', 1:'GBRT', 2:'RF', 
                                         3:'GAM', 4:'MARS', 5:'MLR'})
df_mae = pd.DataFrame(mae).rename(columns={0:'ANN', 1:'GBRT', 2:'RF', 
                                         3:'GAM', 4:'MARS', 5:'MLR'})


# Export model performance results to excel files
df_r2.to_excel('100-fold R2 - All Models.xlsx')
df_mse.to_excel('100-fold MSE - All Models.xlsx')
df_rmse.to_excel('100-fold RMSE - All Models.xlsx')
df_mae.to_excel('100-fold MAE - All Models.xlsx')


print(df_r2['GBRT'].mean()) # Change 'GBRT' to any model to see mean metric
df_r2.head(20) # First 20 rows R^2 dataframe




### ========================= ### HYPERPARAMETER OPTIMIZATION EXAMPLES ### ========================= ###
### ================================== ### ANN OPTIMIZATION ### ==================================== ###

# def ann_architecture(optimizer, activation): # Create ANN model architecture
#     ann_model = Sequential()
#     ann_model.add(Dense(units = 256, input_dim=32, kernel_initializer='normal', activation='relu')) 
#     ann_model.add(Dense(units = 128, kernel_initializer='normal', activation=activation))
#     ann_model.add(Dense(1))
#     ann_model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])    
#     return ann_model

# ann_model = KerasRegressor(build_fn = ann_architecture)

# # GridSearchCV Optimization
# params_ann = {'batch_size': [25,50,75], 
#               'epochs': [75,100,125,150,175], 
#               'optimizer': ['adam', 'nadam'],
#               'activation': ['relu', 'sigmoid']}

# # Fit the model to the hyperparameter grid-search
# ann_opt = GridSearchCV(estimator = ann_model, param_grid=params_ann, 
#                                scoring = 'neg_mean_absolute_error', cv=5, n_jobs=-1)

# ann_opt = ann_opt.fit(X_train, Y_train, verbose=0)

# print(" Results from ANN Grid Search " )
# print("\n The best score across ALL searched params:\n", ann_opt.best_score_)
# print("\n The best parameters across ALL searched params:\n", ann_opt.best_params_)




# ### ========================= ### GBRT OPTIMIZATION ### ========================= ###

# def gbrt_bench(): # Create benchmark GBRT model
#     gbrt_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=2, max_leaf_nodes=5,
#                                            min_samples_leaf=1, min_samples_split=2, random_state=0, loss='squared_error')
#     return gbrt_model

# gbrt_bench = gbrt_bench()

# # Train the benchmark model
# gbrt_bench.fit(X_train, Y_train.ravel())

# # Apply grid optimisation for other hyperparamters here
# params_gbrt = {'n_estimators': [100,200,300,400,500], 
#                'learning_rate': [0.01,0.05,0.1,0.15,0.2,0.25], 
#                'max_depth': [2,4,6,8], 
#                'min_samples_split': [2,4,6,8,10]}

# gbrt_opt = GridSearchCV(estimator=gbrt_bench, param_grid=params_gbrt, cv=5, n_jobs=-1)

# gbrt_opt.fit(X_train, Y_train)

# print(" Results from GBRT Grid Search " )
# print("\n The best estimator across ALL searched params:\n", gbrt_opt.best_estimator_)
# print("\n The best score across ALL searched params:\n", gbrt_opt.best_score_)
# print("\n The best parameters across ALL searched params:\n", gbrt_opt.best_params_)



# ### ========================= ### RF OPTIMIZATION ### ========================= ###

# def rf_bench():
#     rf_model = RandomForestRegressor(n_estimators=500, max_depth=5, max_features=1.0, min_samples_leaf=1, 
#                                      min_samples_split=2, random_state=25, n_jobs=-1, criterion='squared_error')
#     return rf_model

# rf_model = rf_bench()

# # Define grid search parameters
# params_rf = {'n_estimators': [100,200,300,400,500], 
#              'max_features': [1.0, 'sqrt', 'log2'], 
#              'max_depth': [3,4,5,6,7]}

# # Fit the model to the hyperparameter grid-search
# rf_opt = GridSearchCV(estimator=rf_model, param_grid=params_rf, cv=5, n_jobs=-1)
# rf_opt.fit(X_train, Y_train)

# print(" Results from RF Grid Search " )
# print("\n The best estimator across ALL searched params:\n", rf_opt.best_estimator_)
# print("\n The best score across ALL searched params:\n", rf_opt.best_score_)
# print("\n The best parameters across ALL searched params:\n", rf_opt.best_params_)


