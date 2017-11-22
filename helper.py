# -*- coding: utf-8 -*-
"""
Helper for data preprocess and plot

@author: JasonX
"""

import matplotlib.pyplot as plt
import numpy as np

def data_split(X, y, test_size):
    if len(X) != len(y):
        raise SystemError("len(X) != len(y)")
    if not( 0.0 <= test_size < 1):
        raise SystemError("split_size not between 0.0 and 1.0")
    
    total_len = len(X)
    split_idx = int(total_len * (1 - test_size))
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    print("========================")
    print("Original total size = {}".format(total_len))
    print("X_train.shape = {}".format(X_train.shape))
    print("y_train.shape = {}".format(y_train.shape))
    print("X: train size = {}, test size = {}".format(len(X_train), len(X_test)))
    print("y: train size = {}, test size = {}".format(len(y_train), len(y_test)))
    print("========================")
    
    return X_train, X_test, y_train, y_test

def data_preprocess(df, day_shift, date_list, feature_list):

    print("raw data shape = {}".format(df.shape))
    print("raw data df.head(): ")
    print(df.head())
    
    print("DATE_LIST = {}".format(date_list))
    print("FEATURE_LIST = {}".format(feature_list))
    
    extract_list = date_list + feature_list
    df = df[extract_list]
    
    # checking the missing data
    for feature in extract_list:
        if any(df[feature].isnull()):
            print(df[df[feature.isnull]])
        else:
            print("No missing data in feature of '{}'".format(feature))
    
    print("Extracted df.head(): ")
    print(df.head())

    df = df.assign(target = lambda x: x["Close"].shift(day_shift, axis=0))
    df = df[day_shift:].reset_index(drop=True)
    
    # X = np.array(df["Close"])
    X = np.array(df[feature_list])
    y = np.array(df["target"])
    
    # convert 1-D array to 2-D array
    if X.ndim == 1:
        X = X.reshape((len(X), 1))
    if y.ndim == 1:
        y = y.reshape((len(y), 1))
    
    X_train_val, X_test, y_train_val, y_test = data_split(X, y, test_size = 0.1)
    X_train, X_valid, y_train, y_valid = data_split(X_train_val, y_train_val, test_size = 0.2)
    
    X_train_valid_test = (X_train, X_valid, X_test)
    y_train_valid_test = (y_train, y_valid, y_test)
    
    return X_train_valid_test, y_train_valid_test
    
def plot_samples(title, x_label, y_label, data, line_label, images_dir, save_images):

    fig = plt.figure()

    # plt.subplot(122)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range(len(data)):
        x = [idx for idx in range(len(data[i]))]
        plt.plot(x, data[i], label = "{}".format(line_label[i]))

    plt.grid()
    
    plt.legend(loc="best")
    plt.show()
    if save_images:
        fig.savefig(images_dir + str(title) + ".png", dpi=fig.dpi)
    return None

def plot_data(X_train_valid_test, y_train_valid_test, day_shift,
              images_dir, save_images):
    partial_day = 60
    postfix = "_last_" + str(partial_day) + "_days"
    title = ["inputs_" + "training_set" + postfix, 
             "inputs_" + "validation_set" + postfix,
             "inputs_" + "test_set" + postfix]
    x_label = "number of business day"
    y_label = "price"
    line_label = ["input" + "(daily close price)",
                  "target" + "(" + str(day_shift) + " day shift)"]
#==============================================================================
#     line_label = []
#     for i in range(len(feature_list)):
#         line_label.append("input" + "(daily" + feature_list[i] + ")")
#     line_label = 
#                   "target" + "(" + str(day_shift) + " day shift)"]
#==============================================================================
    
    # plot the training/validation/test set    
    for i in range(len(title)):
        # data = [X_train_valid_test[i], y_train_valid_test[i]]
        data = [X_train_valid_test[i][-partial_day:], 
                y_train_valid_test[i][-partial_day:]]
        plot_samples(title[i], x_label, y_label, data, line_label, images_dir, save_images)
    
    return None

def plot_results(y_train, pred_train, y_valid, pred_valid, images_dir, save_images):
    partial_day = 60
    postfix = "_last_" + str(partial_day) + "_days"
    
    title = ["result_" + "training_set" + "_full",
             "result_" + "training_set" + postfix,
             "result_" + "validation_set" + "_full",
             "result_" + "validation_set" + postfix,]
    
    x_label = "number of business day"
    y_label = "price"
    line_label = ["target value", "predition value"]
    
    # title = ["Training set"]
    data_full = [y_train[:], pred_train[:]]
    data_partial = [y_train[-60:], pred_train[-60:]]
    plot_samples(title[0], x_label, y_label, data_full, line_label, images_dir, save_images)
    plot_samples(title[1], x_label, y_label, data_partial, line_label, images_dir, save_images)
    
    # title = ["Validation set"]
    data_full = [y_valid[:], pred_valid[:]]
    data_partial = [y_valid[-partial_day:], pred_valid[-partial_day:]]
    plot_samples(title[2], x_label, y_label, data_full, line_label, images_dir, save_images)
    plot_samples(title[3], x_label, y_label, data_partial, line_label, images_dir, save_images)
    
    return None