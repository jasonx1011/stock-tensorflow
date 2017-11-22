# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:06:34 2017

@author: Chien-Chih Lin
"""

from datetime import datetime
import os
import shutil
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# input raw data
DATA_DIR = "./data/"
CSV_FILE_NAME = "GSPC_1980_2017.csv"
DATA_PATH = os.path.join(DATA_DIR, CSV_FILE_NAME)

# raw data column names
RAW_COLUMN_LIST = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
DATE_LIST = ["Date"]
FEATURE_LIST = ["Close"]

# tensorboard log dir
TF_LOGDIR = "./tf_logs/"

# txt log dir
TXT_LOGDIR = "./txt_logs/"

# images dir
IMAGES_DIR = "./images/"
 
# target (prediction price): number of day shift as target
DAY_SHIFT = 1  

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
    print("X: train size = {}, test size = {}".format(len(X_train), len(X_test)))
    print("y: train size = {}, test size = {}".format(len(y_train), len(y_test)))
    print("========================")
    
    return X_train, X_test, y_train, y_test

def debug_print(x, string):
    print("{} === debug_print ===".format(string))
    print("type ={}, len() = {}".format(type(x), len(x)))
    return None

def plot_samples(title, x_label, y_label, data, line_label):

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
    fig.savefig(IMAGES_DIR + str(title) + ".png", dpi=fig.dpi)
    return None
  
def flatten(x_tensor, name):
    shape = x_tensor.get_shape().as_list()
    # without the 1st param, which is Batch Size
    flatten_dim = np.prod(shape[1:])
    with tf.name_scope(name):
        result = tf.reshape(x_tensor, [-1, flatten_dim], name=name)
        tf.summary.histogram("flatten_layer", result)
    print("flatten")
    print("x_tensor = {}".format(x_tensor))
    print("result = {}".format(result))
    return result

def fully_connect(x_tensor, num_outputs, name):
    """
    Apply a fully connection layers
    """
    shape_list = x_tensor.get_shape().as_list()
    print("fully_connect: ")
    print("input shape = {}".format(shape_list))
    with tf.name_scope(name):
        result = tf.layers.dense(inputs = x_tensor,
                                 units = num_outputs,
                                 activation = tf.nn.relu,
                                 # activation = tf.nn.elu,
                                 kernel_initializer = tf.truncated_normal_initializer(),
                                 name=name)
        tf.summary.histogram("fully_connect_layer", result)
    print("result = {}".format(result))
    return result

def output(x_tensor, num_outputs, name):
    # shape_list = x_tensor.get_shape().as_list()
    
    # linear output activation function: activation = None
    with tf.name_scope(name):
        result = tf.layers.dense(inputs = x_tensor,
                                 units = num_outputs,
                                 activation = None,
                                 kernel_initializer = tf.truncated_normal_initializer(),
                                 name=name)
        tf.summary.histogram("output_layer", result)
    return result

def mlp_net(x, hidden_layers, n_output, name):
    '''
    Multilayer Perceptron: 
    multiple fully connect layers
    '''
    flatten_layer = flatten(x, name="flatten_layer")
    with tf.name_scope(name):
        if len(hidden_layers) == 1:
            fully_layer = fully_connect(flatten_layer, hidden_layers[0],
                                        name="hidden_layer" + "_" + str(0))
        else:
            fully_layer = fully_connect(flatten_layer, hidden_layers[0],
                                        name="hidden_layer" + "_" + str(0))
                
            for layer in range(1, len(hidden_layers)):
                fully_layer = fully_connect(fully_layer, hidden_layers[layer],
                                            name="hidden_layer" + "_" + str(layer))
    with tf.name_scope("prediction"):
        pred = output(fully_layer, n_output, name="output_layer")
        tf.summary.histogram("prediction", pred)
    
    return pred

def next_batch(x, y, batch_size):
    if len(x) != len(y):
        raise SystemError("In def next_batch(x, y), len(x) != len(y)")
    
    idx = 0
    while idx < len(x):
        batch_x = x[idx : idx + batch_size]
        batch_y = y[idx : idx + batch_size]

        yield batch_x, batch_y
        idx += batch_size
        
def clean_files(rm_path, bak_path):
    '''
    make a backup and remove all the files
    '''
    if not os.path.isdir(rm_path):
        os.makedirs(rm_path)
    if not os.path.isdir(bak_path):
        os.makedirs(bak_path)

    shutil.rmtree(bak_path)
    shutil.copytree(rm_path, bak_path)
    shutil.rmtree(rm_path)
    
    os.makedirs(rm_path)

    return None
    
def run_tf(log_timestr, X_train_valid_test, y_train_valid_test, turn_on_tf_board,
           lr, epochs, batch_size, hidden_layers, hparam_str):
    
    # unpack input data
    X_train, X_valid, X_test = X_train_valid_test
    y_train, y_valid, y_test = y_train_valid_test
    
    # number of input layer and output layer
    n_input = 1
    n_output = 1


    tf.reset_default_graph()

    # tf Graph input
    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, n_input], name = "x")
        y = tf.placeholder(tf.float32, [None, n_output], name = "y")
    
    # Model
    # flattern x inside the mlp_net function
    pred = mlp_net(x, hidden_layers, n_output, name="mlp_net")
     
    # Loss and Optimizer
    with tf.name_scope("cost"):
        # mse
        # cost = tf.reduce_mean(tf.squared_difference(pred, y))
        # rmse
        cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(pred, y)))
        tf.summary.scalar("cost", cost)
        # for validation set
        # rmse_valid = tf.sqrt(tf.reduce_mean(tf.squared_difference(pred, y)))
        # tf.summary.scalar("rmse_valid", rmse_valid)
    
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    
    merged = tf.summary.merge_all()
    
    start_time = timeit.default_timer()
    
    # Launch the graph
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
        
        if turn_on_tf_board:
            train_writer = tf.summary.FileWriter(TF_LOGDIR + hparam_str + "/train/", sess.graph)
            valid_writer = tf.summary.FileWriter(TF_LOGDIR + hparam_str + "/valid/")

        # Training cycle
        print('Training...')
        for epoch in range(epochs):
#==============================================================================
#             n_batches = int(len(X_train)/batch_size)
#==============================================================================
            
            # Loop over all batches
            batch_count = 0
            for batch_x, batch_y in next_batch(X_train, y_train, batch_size):
                
                sess.run(optimizer, feed_dict={x: batch_x,
                                               y: batch_y})
#==============================================================================
#                 loss = sess.run(cost, feed_dict = {x: batch_x,
#                                                    y: batch_y})
#==============================================================================

#==============================================================================
#                 if batch_count % (n_batches // 10) == 0:
#                     print("batch_count = {}".format(batch_count))
#                     print("batch training loss = {:>5.3f}".format(loss))
#==============================================================================
                
                batch_count += 1

            print("Epoch {:>2} :".format(epoch))
            
            with tf.name_scope("total_loss"):
                loss_train, summary_train = sess.run([cost, merged], 
                                                     feed_dict={x: X_train, y: y_train})
                # tf.summary.scalar("loss_train", loss_train)
                loss_valid, summary_valid = sess.run([cost, merged],
                                                     feed_dict={x: X_valid, y: y_valid})                
                # tf.summary.scalar("loss_valid", loss_valid)
                                    
            print("epoch training loss = {:>5.3f}".format(loss_train))
            print("epoch validation loss = {:>5.3f}".format(loss_valid))
            if turn_on_tf_board:
                train_writer.add_summary(summary_train, epoch)
                valid_writer.add_summary(summary_valid, epoch)

        print("Optimization Finished!")
        
        pred_train = sess.run(pred , feed_dict = {x: X_train,
                                                  y: y_train})
        pred_valid = sess.run(pred , feed_dict = {x: X_valid,
                                                  y: y_valid})
                                                  
        plot_results(y_train, pred_train, y_valid, pred_valid)
        
        print("==================================================")
        print("Final total_loss: \n")
        print("loss_train = {:>5.3f}".format(loss_train))
        print("loss_valid = {:>5.3f}".format(loss_valid))
        print("==================================================")
        
    end_time = timeit.default_timer()
    runtim_min = (end_time - start_time)/60
    print("runtime = {:.3f} (mins)".format(runtim_min))
    
    with open(TXT_LOGDIR + "log_" + log_timestr + ".txt", "a") as logfile:
        logfile.write("==================================================\n")
        logfile.write("hyper parameter: \n")
        logfile.write("\n")
        logfile.write("[learning rate, epochs, batch_size, hidden_layers]\n")
        logfile.write("{}\n".format(hparam_str))
        logfile.write("\n")
        logfile.write("Final total_loss:\n")
        logfile.write("loss_train = {:>5.3f} \n".format(loss_train))
        logfile.write("loss_valid = {:>5.3f} \n".format(loss_valid))
        logfile.write("\n")
        logfile.write("runtime = {:.3f} (mins)\n".format(runtim_min))
        logfile.write("==================================================\n")
    # Save Model
    # saver = tf.train.Saver()
    # save_path = saver.save(sess, save_model_path)
    
    return None

def data_preprocess(df, day_shift):

    print("raw data shape = {}".format(df.shape))
    print("raw data df.head(): ")
    print(df.head())
    
    print("DATE_LIST = {}".format(DATE_LIST))
    print("FEATURE_LIST = {}".format(FEATURE_LIST))
    
    extract_list = DATE_LIST + FEATURE_LIST
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
    
    X = np.array(df["Close"])
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

def plot_data(X_train_valid_test, y_train_valid_test, day_shift):
    partial_day = 60
    postfix = "_last_" + str(partial_day) + "_days"
    title = ["inputs_" + "training_set" + postfix, 
             "inputs_" + "validation_set" + postfix,
             "inputs_" + "test_set" + postfix]
    x_label = "number of business day"
    y_label = "price"
    line_label = ["input" + "(daily close price)",
                  "target" + "(" + str(day_shift) + " day shift)"]
    
    # plot the training/validation/test set    
    for i in range(len(title)):
        # data = [X_train_valid_test[i], y_train_valid_test[i]]
        data = [X_train_valid_test[i][-partial_day:], 
                y_train_valid_test[i][-partial_day:]]
        plot_samples(title[i], x_label, y_label, data, line_label)
    
    return None

def plot_results(y_train, pred_train, y_valid, pred_valid):
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
    plot_samples(title[0], x_label, y_label, data_full, line_label)
    plot_samples(title[1], x_label, y_label, data_partial, line_label)
    
    # title = ["Validation set"]
    data_full = [y_valid[:], pred_valid[:]]
    data_partial = [y_valid[-partial_day:], pred_valid[-partial_day:]]
    plot_samples(title[2], x_label, y_label, data_full, line_label)
    plot_samples(title[3], x_label, y_label, data_partial, line_label)
    
    return None

def make_hparam_string(lr, epochs, batch_size, hidden_layers):
    
    hparam_str = "lr_{:.0E},{},{},h_layers".format(lr, epochs, batch_size)
    idx = 0
    while idx < len(hidden_layers):
        hparam_str = hparam_str + "_" + str(hidden_layers[idx])
        idx += 1
    return hparam_str

def backup_files():
        clean_files(TF_LOGDIR, os.path.join("backup", TF_LOGDIR))
        clean_files(TXT_LOGDIR, os.path.join("backup", TXT_LOGDIR))
        clean_files(IMAGES_DIR, os.path.join("backup", IMAGES_DIR))

def main():
    turn_on_tf_board = True
    # turn_on_txt_log = True
    # save_images = True  
    
    backup_files()
        
    df = pd.read_csv(DATA_PATH)
    X_train_valid_test, y_train_valid_test = data_preprocess(df, DAY_SHIFT)
    
    plot_data(X_train_valid_test, y_train_valid_test, DAY_SHIFT)
        
    log_timestr = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    with open(TXT_LOGDIR + "log_" + log_timestr + ".txt", "w") as logfile:
        logfile.close()
        
    # Hyper parameters default values
    # after experiments, the good hyper parameters setting:
    # lr = 1E-3, epochs = 10, batch_size = 128, hidden_layers = [16]
    
    lr = 1E-3
    epochs = 10
    batch_size = 128
    # hidden_layers = [256, 128]
    hidden_layers = [16]
    
    hparam_str = make_hparam_string(lr, epochs, batch_size, hidden_layers)
    print(hparam_str)
    
    run_tf(log_timestr, X_train_valid_test, y_train_valid_test, turn_on_tf_board,
           lr, epochs, batch_size, hidden_layers, hparam_str)
    
#==============================================================================
#     for lr in [1E-3, 1E-4]:
#         for batch_size in [1, 32, 128, 256]:
#             for layer_1 in [1, 16, 32, 128]:
#                 hidden_layers = [layer_1]
#                 hparam_str = make_hparam_string(lr, epochs, 
#                                                 batch_size, hidden_layers)
#                 
#                 run_tf(log_timestr, X_train_valid_test, y_train_valid_test, 
#                        turn_on_tf_board,
#                        lr, epochs, batch_size, hidden_layers, hparam_str)
#                 
#                 for layer_2 in [1, 16, 32, 128]:
#                     hidden_layers = [layer_1, layer_2]
#                     hparam_str = make_hparam_string(lr, epochs, 
#                                                     batch_size, hidden_layers)
#                     
#                     run_tf(log_timestr, X_train_valid_test, y_train_valid_test,
#                            turn_on_tf_board,
#                            lr, epochs, batch_size, hidden_layers, hparam_str)
#==============================================================================

    return None

if __name__ == "__main__":
    main()
