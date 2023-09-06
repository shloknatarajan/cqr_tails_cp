import os
import sys
sys.path.append('../')
import torch
import random
import numpy as np
import pandas as pd
import helper
from datasets import datasets
from sklearn import linear_model
from nonconformist.nc import NcFactory
from nonconformist.nc import RegressorNc
from nonconformist.nc import AbsErrorErrFunc
from nonconformist.nc import QuantileRegErrFunc
from nonconformist.nc import RegressorNormalizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from nonconformist.nc import QuantileRegAsymmetricErrFunc
from sklearn.datasets import load_diabetes
import pickle
import datetime
# pd.set_option('float_format', '{:.3f}')

    

def run_experiment(dataset_name,
                   random_state_train_test,
                   save_to_csv=True):
    """ Estimate prediction intervals and print the average length and coverage

    Parameters
    ----------

    dataset_name : array of strings, list of datasets
    test_method  : string, method to be tested, estimating
                   the 90% prediction interval
    random_state_train_test : integer, random seed to be used
    save_to_csv : boolean, save average length and coverage to csv (True)
                  or not (False)

    """
    print(f"dataset: {dataset_name}")
    dataset_name_vec = []
    method_vec = []
    coverage_vec = []
    length_vec = []
    seed_vec = []

    seed = random_state_train_test
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    coverage_cp_qnet=0
    length_cp_qnet=0

    


    # determines the size of test set
    test_ratio = 0.2

    # conformal prediction miscoverage level
    significance = 0.1
    # desired quantile levels, used by the quantile regression methods
    quantiles = [0.05, 0.95]

    # Random forests parameters (shared by conditional quantile random forests
    # and conditional mean random forests regression).
    n_estimators = 1000 # usual random forests n_estimators parameter
    min_samples_leaf = 1 # default parameter of sklearn

    # Quantile random forests parameters.
    # See QuantileForestRegressorAdapter class for more details
    quantiles_forest = [5, 95]
    CV_qforest = True
    coverage_factor = 0.85
    cv_test_ratio = 0.05
    cv_random_state = 1
    cv_range_vals = 30
    cv_num_vals = 10

    # Neural network parameters  (shared by conditional quantile neural network
    # and conditional mean neural network regression)
    # See AllQNet_RegressorAdapter and MSENet_RegressorAdapter in helper.py
    nn_learn_func = torch.optim.Adam
    epochs = 1000
    lr = 0.0005
    hidden_size = 64
    batch_size = 64
    dropout = 0.1
    wd = 1e-6

    # Ask for a reduced coverage when tuning the network parameters by
    # cross-validation to avoid too conservative initial estimation of the
    # prediction interval. This estimation will be conformalized by CQR.
    quantiles_net = [0.1, 0.9]


    # local conformal prediction parameter.
    # See RegressorNc class for more details.
    beta = 1
    beta_net = 1

    # local conformal prediction parameter. The local ridge regression method
    # uses nearest neighbor regression as the MAD estimator.
    # Number of neighbors used by nearest neighbor regression.
    n_neighbors = 11
    X = None
    y = None

    # X, y = load_diabetes(return_X_y=True)
    # y = (y - y.mean()) / y.std()
    # X = X / np.linalg.norm(X)
    base_dataset_path = '/Users/shloknatarajan/vi_conformal_bayes/cqr/datasets/'
    X, y = datasets.GetDataset(dataset_name, base_dataset_path)

    # Dataset is divided into test and train data based on test_ratio parameter
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_ratio,
                                                        random_state=random_state_train_test)


    # fit a simple ridge regression model (sanity check)
    model = linear_model.RidgeCV()
    model = model.fit(X_train, np.squeeze(y_train))
    predicted_data = model.predict(X_test).astype(np.float32)

    # calculate the normalized mean squared error
    print("Ridge relative error: %f" % (np.sum((np.squeeze(y_test)-predicted_data)**2)/np.sum(np.squeeze(y_test)**2)))
    sys.stdout.flush()

    # reshape the data
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    # input dimensions
    n_train = X_train.shape[0]
    in_shape = X_train.shape[1]

    print("Size: train (%d, %d), test (%d, %d)" % (X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1]))
    sys.stdout.flush()

    # set seed for splitting the data into proper train and calibration
    np.random.seed(seed)
    idx = np.random.permutation(n_train)

    # divide the data into proper training set and calibration set
    n_half = int(np.floor(n_train/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]
    
    # zero mean and unit variance scaling of the train and test features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X_train[idx_train])
    X_train = scalerX.transform(X_train)
    X_test = scalerX.transform(X_test)
    
    # scale the labels by dividing each by the mean absolute response
    mean_ytrain = np.mean(np.abs(y_train[idx_train]))
    y_train = np.squeeze(y_train)/mean_ytrain
    y_test = np.squeeze(y_test)/mean_ytrain

    ##### Removed the if statement, only using cqr net
    model = helper.AllQNet_RegressorAdapter(model=None,
                                            fit_params=None,
                                            in_shape = in_shape,
                                            hidden_size = hidden_size,
                                            quantiles = quantiles_net,
                                            learn_func = nn_learn_func,
                                            epochs = epochs,
                                            batch_size=batch_size,
                                            dropout=dropout,
                                            lr=lr,
                                            wd=wd,
                                            test_ratio=cv_test_ratio,
                                            random_state=cv_random_state,
                                            use_rearrangement=False)
    nc = RegressorNc(model, QuantileRegErrFunc())

    y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)

    helper.plot_func_data(y_test,y_lower,y_upper, f"CQR Net: {dataset_name}")
    coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(y_test,y_lower,y_upper,significance,"CQR Net")


    dataset_name_vec.append(dataset_name)
    method_vec.append('CQR Net')
    coverage_vec.append(coverage_cp_qnet)
    length_vec.append(length_cp_qnet)
    seed_vec.append(seed)

    ############### Summary

    coverage_str = 'Coverage (expected ' + str(100 - significance*100) + '%)'
    results = np.array([[dataset_name, coverage_str, 'Avg. Length', 'Seed'],
                     ['CP Quantile Net', coverage_cp_qnet, length_cp_qnet, seed]])

    results_ = pd.DataFrame(data=results[1:,1:],
                      index=results[1:,0],
                      columns=results[0,1:])

    print("== SUMMARY == ")
    print("dataset name: " + dataset_name)
    print(results_)
    sys.stdout.flush()
    dt = datetime.datetime.now()
    str_dt = dt.strftime('%Y-%m-%d-%H:%M:%S')
    with open(f'cqr_saved_models/{dataset_name}-{str_dt}.pkl', 'wb') as outfile:
        pickle.dump(nc, outfile)

    if save_to_csv:
        results = pd.DataFrame(results)

        outdir = './results/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        out_name = outdir + 'results.csv'

        df = pd.DataFrame({'name': dataset_name_vec,
                           'method': method_vec,
                           coverage_str : coverage_vec,
                           'Avg. Length' : length_vec,
                           'seed': seed_vec})

        if os.path.isfile(out_name):
            df2 = pd.read_csv(out_name)
            df = pd.concat([df2, df], ignore_index=True)

        df.to_csv(out_name, index=False)


# Parameters
# ----------

# dataset_name : array of strings, list of datasets
# test_method  : string, method to be tested, estimating
#                the 90% prediction interval
# random_state_train_test : integer, random seed to be used
# save_to_csv : boolean, save average length and coverage to csv (True)
#               or not (False)
dataset_name = 'concrete'
random_state_train_test = 1
save_to_csv = True
run_experiment(dataset_name, random_state_train_test, save_to_csv)