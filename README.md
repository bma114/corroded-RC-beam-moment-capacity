# Predicting the Residual Bending Moment Capacity of Corroded RC Beams

## 1. Project Description

This is an example implementation of six predictive machine learning models used to estimate the residual moment capacity of corroded reinforced concrete (RC) beams failing in flexural bending. The models implement an extensive database of 804 monotonic bending tests and 32 input features, collected from 54 experimental campaigns available in the literature. The models included in this repository are:

* Artificial Neural Network (ANN).
* Gradient Boosting Regression Trees (GBRT).
* Random Forest (RF).
* Multivariate Adaptive Regression Splines (MARS).
* Generalized Additive Model (GAM).
* Multiple Linear Regression (MLR).

## 2. Database

The code presented in this repository focuses on the prediction of a single response variable (residual bending moment capacity) and is taken from a larger study predicting five key mechanical properties of corroded RC beams. The article manuscript is currently under review and will be attached once available. Data.csv provides all the necessary data to execute the run.py script. The complete database is available open-source at: https://zenodo.org/records/8062007 

## 3. Model Training

Each model is trained and tested using a Repeated k-fold cross-validation approach. A 10-fold split is implemented and repeated ten times, representing a 100-fold approach, with each dataset randomly reshuffled between repetitions.

## 4. Instructions

Follow the instructions below to execute the script and build the models:
1.	Download the zip file containing all files in the repository to your local drive. 
2.	Extract or unzip the folder, keeping all files together without moving the relative path between them. 
3.	Using a Python environment of your choice (e.g., Jupyter Notebook, Visual Studio Code, Spyder, etc.), open the run.py file.
4.	Check that all Python dependencies required to run the script have been installed in your Python environment. See the list below for all the necessary packages. 
5.	Once all the necessary packages are installed, execute the run.py script to train and test the models. 
6.	Note that due to the extensive Repeated K-fold training algorithm adopted in this study, the script will likely take between 30 minutes to 1 hour to run entirely, depending on the CPU of the local device. 

## 5. Code Structure

The run.py file is organized in the following format:

* Data Preparation.
* Model building, repeated K-fold cross-validation, and predictions.
* Model performance and error evaluation.
* Hyperparameter optimization.

Functions.py includes additional functions required to execute the run.py file.

Because of the considerable computational expense of running 100 cross-validation iterations for nine different models, hyperparameter optimization has been excluded from the training loop. Example code for hyperparameter optimization is provided at the end of run.py.

The run.py script outputs a 9x100 dataframe compiling each model’s performance, evaluated against the R2, MSE, RMSE, and MAE. 

## 6. Related Work

A similar study investigating the mechanical degradation of corroded reinforcing steel can be found at: https://github.com/bma114/corroded-steel-machine-learning 

With a published journal article, which can be accessed at: https://doi.org/10.1016/j.conbuildmat.2024.137023 

The complete open-source database is available at: https://zenodo.org/records/8035720 

## 7. Dependencies

The application includes the following dependencies to run:
* Python == 3.11.0
* Pandas == 1.4.4
* NumPy == 1.26.4
* TensorFlow == 2.16.1
* Keras == 3.3.3
* Scikit-Learn == 1.3.2
* PyGAM == 0.9.1
* PyEarth == 0.1.25

