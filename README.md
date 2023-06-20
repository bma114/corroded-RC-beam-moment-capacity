# Predicting the Residual Bending Moment Capacity of Corroded RC Beams

## Project Description

This is an example implementation of six predictive machine learning models used to estimate the residual moment capacity of corroded reinforced concrete (RC) beams failing in flexural bending. The models implement an extensive database of 804 monotonic bending tests and 32 input features, collected from 54 experimental campaigns available in the literature. The models included in this repository are:

* Artificial Neural Network (ANN).
* Gradient Boosting Regression Trees (GBRT).
* Random Forest (RF).
* Multivariate Adaptive Regression Splines (MARS).
* Generalized Additive Model (GAM).
* Multiple Linear Regression (MLR).

## Database

The code presented in this repository focuses on the prediction of a single response variable (residual bending moment capacity) and is taken from a larger study predicting five key mechanical properties of corroded RC beams. The article manuscript is currently under review and will be attached once available. Data.csv provides all the necessary data to execute the run.py script. The complete database is available open-source at the DOI: 10.5281/zenodo.8062007.

## Code Structure

The run.py script outputs a 9x100 dataframe compiling each model’s performance, evaluated against the R2, MSE, RMSE, and MAE. The prediction regression and residual errors of the best-performing model are also plotted at the end of the file. 

Functions.py includes additional functions necessary for the run.py execution.

Run.py is organized in the following format:

* Data Preparation
* Model building and implementation using a 100-fold cross-validation training method.
* Model performance and error evaluation.
* Plotting
* Hyperparameter optimization examples.

**Note:** Because of the considerable computational expense in running 100 cross-validation iterations, hyperparameter optimization is mostly excluded from the governing training loop. Example code for hyperparameter optimization is provided at the end of run.py for those models not optimized within the cross-validation training loop.

## Related Work

A similar study investigating the mechanical degradation of corroded reinforcing steel can be found at: https://github.com/bma114/corroded-steel-machine-learning

With the complete open-source database available at the DOI: 10.5281/zenodo.8035720.
