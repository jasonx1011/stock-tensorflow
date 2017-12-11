# -*- coding: utf-8 -*-
"""
Goal:
Using S&P500 historical data from yahoo finance to 
predict S&P500 price for the next day

@author: Chien-Chih Lin
"""

from datetime import datetime
import os
import timeit

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import tensorflow as tf

import file_manager
import helper

# input raw data
DATA_DIR = "./data/"
CSV_FILE_NAME = "GSPC_1980_2017.csv"
DATA_PATH = os.path.join(DATA_DIR, CSV_FILE_NAME)

# raw data column names
RAW_COLUMN_LIST = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
DATE_LIST = ["Date"]
# FEATURE_LIST = ["Close"]
# we could not pick both "Open" & "Close",
# since the next day "Open" == the "Close" of today 
# the target information leak
# FEATURE_LIST = ["Open", "High", "Low", "Close"]
FEATURE_LIST = ["High", "Low", "Close"]

# tensorboard log dir
TF_LOGDIR = "./tf_logs/"

# txt log dir
TXT_LOGDIR = "./txt_logs/"

# images dir
IMAGES_DIR = "./images/"
 
# target (prediction price): number of day shift as target
DAY_SHIFT = 1  

def debug_print(x, string):
    print("{} === debug_print ===".format(string))
    print("type ={}, len() = {}".format(type(x), len(x)))
    return None

def flatten(x_tensor, name):
    """
    Flatten input layer
    """
    # without the 1st param, which is Batch Size
    shape = x_tensor.get_shape().as_list()
    flatten_dim = np.prod(shape[1:])
    with tf.name_scope(name):
        result = tf.reshape(x_tensor, [-1, flatten_dim], name=name)
        tf.summary.histogram("flatten_layer", result)
    print("==================================================")
    print("flatten:")
    print("input x_tensor = {}".format(x_tensor))
    print("result = {}".format(result))
    print("==================================================")
    return result

def fully_connect(x_tensor, num_outputs, name):
    """
    Apply a fully connection layer
    """
    # shape_list = x_tensor.get_shape().as_list()
    with tf.name_scope(name):
        result = tf.layers.dense(inputs = x_tensor,
                                 units = num_outputs,
                                 activation = tf.nn.relu,
                                 # activation = tf.nn.elu,
                                 kernel_initializer = tf.truncated_normal_initializer(),
                                 name=name)
        tf.summary.histogram("fully_connect_layer", result)
    print("==================================================")
    print("fully_connect:")
    print("input x_tensor = {}".format(x_tensor))
    print("result = {}".format(result))
    print("==================================================")
    return result

def output(x_tensor, num_outputs, name):
    """
    Apply a output layer with linear output activation function
    """
    # shape_list = x_tensor.get_shape().as_list()
    # linear output activation function: activation = None
    with tf.name_scope(name):
        result = tf.layers.dense(inputs = x_tensor,
                                 units = num_outputs,
                                 activation = None,
                                 kernel_initializer = tf.truncated_normal_initializer(),
                                 name=name)
        tf.summary.histogram("output_layer", result)
    print("==================================================")
    print("output:")
    print("input x_tensor = {}".format(x_tensor))
    print("result = {}".format(result))
    print("==================================================")
    return result

def mlp_net(x, hidden_layers, n_output, name):
    """
    Multilayer Perceptron: 
    multiple fully connect layers
    """
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

def next_batch(x, y, batch_size, shuffle_flag):
    if len(x) != len(y):
        raise SystemError("In def next_batch(x, y), len(x) != len(y)")
    
    idx = 0
    while idx < len(x):
        batch_x = x[idx : idx + batch_size]
        batch_y = y[idx : idx + batch_size]
#==============================================================================
#         print("orig:")
#         print(batch_x[:3])
#         print(batch_y[:3])
#==============================================================================
        if shuffle_flag:
            a, b = shuffle(batch_x, batch_y)
#==============================================================================
#             print("shuffle:")
#             print(a[:3])
#             print(b[:3])
#==============================================================================
            yield a, b
        else:
            yield batch_x, batch_y
        idx += batch_size

def scale(X_orig, y_orig):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(X_orig)
    scaler_y.fit(y_orig)
    X_scaled = scaler_X.transform(X_orig)
    y_scaled = scaler_y.transform(y_orig)
#==============================================================================
#     verify the scaling
#==============================================================================
#==============================================================================
#     X_recover = np.multiply(X_scaled, scaler_X.data_range_) + scaler_X.data_min
#     y_recover = np.multiply(y_scaled, scaler_y.data_range_) + scaler_y.data_min
#     print("x data range:")
#     print(scaler_X.data_min_)
#     print(scaler_X.data_max_)
#     print(scaler_X.data_range_)
#     print("y data range:")
#     print("min".format(scaler_y.data_min_))
#     print("max".format(scaler_y.data_max_))
#     print("range".format(scaler_y.data_range_))
#     
#     print("X_orig")
#     print(X_orig)
#     print("X_recover")
#     print(X_recover)
#     print("y_orig")
#     print(y_orig)
#     print("y_recover")
#     print(y_recover)
#     print(X_scaled[3:])
#     print(y_scaled[3:])
#==============================================================================
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def recover(scaled, scaler):
    recovered_data = np.multiply(scaled, scaler.data_range_) + scaler.data_min
    return recovered_data

def run_tf(log_timestr, X_train_valid_test, y_train_valid_test, turn_on_tf_board,
           lr, epochs, batch_size, hidden_layers, hparam_str, save_images):
    
    # unpack input data
    X_train, X_valid, X_test = X_train_valid_test
    y_train, y_valid, y_test = y_train_valid_test
    
    X_train, y_train, scaler_X_train, scaler_y_train = scale(X_train, y_train)
    X_valid, y_valid, scaler_X_valid, scaler_y_valid = scale(X_valid, y_valid)
    X_test, y_test, scaler_X_test, scaler_y_test = scale(X_train, y_train)
    
    shuffle_flag = True
    if shuffle_flag:
        X_train, y_train = shuffle(X_train, y_train)
    # number of input layer and output layer
    # n_input = 1
    # n_output = 1
    n_input = X_train.shape[1]
    n_output = y_train.shape[1]

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
        # optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
        optimizer = tf.train.AdamOptimizer(3E-4).minimize(cost)
        # optimizer = tf.train.AdagradOptimizer(lr).minimize(cost)
    
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
            for batch_x, batch_y in next_batch(X_train, y_train, batch_size, shuffle_flag=shuffle_flag):
                
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

            
            with tf.name_scope("total_loss"):
                loss_train, summary_train = sess.run([cost, merged], 
                                                     feed_dict={x: X_train, y: y_train})
                # tf.summary.scalar("loss_train", loss_train)
                loss_valid, summary_valid = sess.run([cost, merged],
                                                     feed_dict={x: X_valid, y: y_valid})                
                # tf.summary.scalar("loss_valid", loss_valid)
            
            if (epochs // 10) > 0 and epoch % (epochs // 10) == 0:
                print("Epoch {:>2} :".format(epoch))
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
                                                  
        helper.plot_results(y_train, pred_train, y_valid, pred_valid,
                            IMAGES_DIR, save_images)
                            
        pred_train = recover(pred_train, scaler_y_train)
        y_train = recover(y_train, scaler_y_train)
        
        pred_valid = recover(pred_valid, scaler_y_valid)
        y_valid = recover(y_valid, scaler_y_valid)
        
        diff_train = pred_train - y_train
        diff_valid = pred_valid - y_valid

        error_train = np.mean(np.abs(diff_train/y_train))
        error_valid = np.mean(np.abs(diff_valid/y_valid))
        
        max_diff_train, min_diff_train = float(max(diff_train)), float(min(diff_train))
        max_diff_valid, min_diff_valid = float(max(diff_valid)), float(min(diff_valid))
                
        rmse_train = np.sqrt(np.mean((diff_train)**2))
        rmse_valid = np.sqrt(np.mean((diff_valid)**2))
        
        print("==================================================")
        print("Final total_loss: \n")
        print("rmse_train = {:>5.3f}, {}".format(rmse_train, hparam_str))
        print("error_train = {:>5.3f}, {}".format(error_train, hparam_str))
        print("max/min difference: {:5.3f} to {:5.3f}".format(max_diff_train, min_diff_train))
        print("")                                                        
        print("rmse_valid = {:>5.3f}, {}".format(rmse_valid, hparam_str))
        print("error_valid = {:>5.3f}, {}".format(error_valid, hparam_str))        
        print("max/min difference: {:5.3f} to {:5.3f}".format(max_diff_valid, min_diff_valid))
        print("==================================================")
    
    end_time = timeit.default_timer()
    runtim_min = (end_time - start_time)/60
    print("runtime = {:.3f} (mins)".format(runtim_min))
    print("==================================================")
    
    with open(TXT_LOGDIR + "log_" + log_timestr + ".txt", "a") as logfile:
        logfile.write("==================================================\n")
        logfile.write("hyper parameter:\n")
        logfile.write("\n")
        logfile.write("[learning rate, epochs, batch_size, hidden_layers]\n")
        logfile.write("{}\n".format(hparam_str))
        logfile.write("\n")
        logfile.write("Final total_loss:\n")
        logfile.write("rmse_train = {:>5.3f}, {}\n".format(rmse_train, hparam_str))
        logfile.write("error_train = {:>5.3f}, {}\n".format(error_train, hparam_str))
        logfile.write("max/min difference: {:5.3f} to {:5.3f}\n".format(max_diff_train, min_diff_train))
        logfile.write("\n")                                                        
        logfile.write("rmse_valid = {:>5.3f}, {}\n".format(rmse_valid, hparam_str))
        logfile.write("max/min difference: {:5.3f} to {:5.3f}\n".format(max_diff_valid, min_diff_valid))
        logfile.write("error_valid = {:>5.3f}, {}\n".format(error_valid, hparam_str))        
        logfile.write("\n")
        logfile.write("runtime = {:.3f} (mins)\n".format(runtim_min))
        logfile.write("==================================================\n")
    
    # Save Model
    # saver = tf.train.Saver()
    # save_path = saver.save(sess, save_model_path)
    
    return None

def make_hparam_string(lr, epochs, batch_size, hidden_layers):
    
    hparam_str = "lr_{:.0E},{},{},h_layers".format(lr, epochs, batch_size)
    idx = 0
    while idx < len(hidden_layers):
        hparam_str = hparam_str + "_" + str(hidden_layers[idx])
        idx += 1
    return hparam_str
 
def avg_model(log_timestr, X_train_valid_test, y_train_valid_test):
    # unpack input data
    X_train, X_valid, X_test = X_train_valid_test
    y_train, y_valid, y_test = y_train_valid_test
    
    # use the mean() of features as prediction
    # print(X_valid)
    pred_train = np.mean(X_train[:], axis=1)
    pred_valid = np.mean(X_valid[:], axis=1)
    
    pred_train = pred_train.reshape(y_train.shape)
    pred_valid = pred_valid.reshape(y_valid.shape)
    # print(pred_valid)
    # print(y_valid)
    
    # rmse loss
    diff_train = pred_train - y_train
    diff_valid = pred_valid - y_valid
    
    error_train = np.mean(np.abs(diff_train/y_train))
    error_valid = np.mean(np.abs(diff_valid/y_valid))
    
    max_diff_train, min_diff_train = float(max(diff_train)), float(min(diff_train))
    max_diff_valid, min_diff_valid = float(max(diff_valid)), float(min(diff_valid))
            
    rmse_train = np.sqrt(np.mean((diff_train)**2))
    rmse_valid = np.sqrt(np.mean((diff_valid)**2))
    
    print("##################################################")
    print("Avg. model: \n")
    print("rmse_train = {:>5.3f}".format(rmse_train))
    print("error_train = {:>5.3f}".format(error_train))
    print("max/min difference: {:5.3f} to {:5.3f}".format(max_diff_train, min_diff_train))
    print("")                                                        
    print("rmse_valid = {:>5.3f}".format(rmse_valid))
    print("error_valid = {:>5.3f}".format(error_valid))
    print("max/min difference: {:5.3f} to {:5.3f}".format(max_diff_valid, min_diff_valid))
    print("##################################################")
    
    with open(TXT_LOGDIR + "log_" + log_timestr + ".txt", "a") as logfile:
        logfile.write("##################################################\n")
        logfile.write("\n")
        logfile.write("Avg. model:\n")
        logfile.write("rmse_train = {:>5.3f}, {}\n".format(rmse_train, "Avg. model"))
        logfile.write("error_train = {:>5.3f}, {}\n".format(error_train, "Avg. model"))
        logfile.write("max/min difference: {:5.3f} to {:5.3f}\n".format(max_diff_train, min_diff_train))
        logfile.write("\n")                                                        
        logfile.write("rmse_valid = {:>5.3f}, {}\n".format(rmse_valid, "Avg. model"))
        logfile.write("error_valid = {:>5.3f}, {}\n".format(error_valid, "Avg. model"))
        logfile.write("max/min difference: {:5.3f} to {:5.3f}\n".format(max_diff_valid, min_diff_valid))
        logfile.write("##################################################\n")    
    return None
   
def main():
    
    turn_on_tf_board = True
    # turn_on_txt_log = True
    save_images = False  
    single_run = True
    # single_run = False
    
    file_manager.backup_files([TF_LOGDIR, TXT_LOGDIR, IMAGES_DIR])
    
    log_timestr = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    with open(TXT_LOGDIR + "log_" + log_timestr + ".txt", "w") as logfile:
        logfile.close()
    
    df = pd.read_csv(DATA_PATH)
    X_train_valid_test, y_train_valid_test = helper.data_preprocess(df,
                                                                    DAY_SHIFT,
                                                                    DATE_LIST,
                                                                    FEATURE_LIST)
    
    helper.plot_data(X_train_valid_test, y_train_valid_test,
                     DAY_SHIFT, IMAGES_DIR, save_images, FEATURE_LIST)
                     
    # Hyper parameters default values
    lr = 3E-3
    epochs = 2000
    batch_size = 128
    # batch_size = 32
    hidden_layers = [16, 32]
    # hidden_layers = [16, 32, 16, 8]
    # hidden_layers = [64, 64, 32, 16]
    # hidden_layers = [16]
    
    hparam_str = make_hparam_string(lr, epochs, batch_size, hidden_layers)
    print(hparam_str)
    
    if single_run:
        run_tf(log_timestr, X_train_valid_test, y_train_valid_test, 
               turn_on_tf_board,
               lr, epochs, batch_size, hidden_layers, hparam_str, save_images)
    else:
        for lr in [3E-3, 1E-4]:
            for batch_size in [32, 128]:
                for layer_1 in [16, 32, 64]:
                    hidden_layers = [layer_1]
                    hparam_str = make_hparam_string(lr, epochs, 
                                                    batch_size, hidden_layers)
                    
                    run_tf(log_timestr, X_train_valid_test, y_train_valid_test, 
                           turn_on_tf_board,
                           lr, epochs, batch_size, hidden_layers, 
                           hparam_str, save_images)
                    
                    for layer_2 in [16, 32, 64]:
                        hidden_layers = [layer_1, layer_2]
                        hparam_str = make_hparam_string(lr, epochs, 
                                                        batch_size, hidden_layers)
                        
                        run_tf(log_timestr, X_train_valid_test, y_train_valid_test,
                               turn_on_tf_board,
                               lr, epochs, batch_size, hidden_layers,
                               hparam_str, save_images)
    
#==============================================================================
#     X_train_valid_test, y_train_valid_test = helper.data_preprocess(df,
#                                                                     DAY_SHIFT,
#                                                                     DATE_LIST,
#                                                                     ["Close"])
#==============================================================================

    avg_model(log_timestr, X_train_valid_test, y_train_valid_test)
    
    return None

if __name__ == "__main__":
    main()
