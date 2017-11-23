# stock-tensorflow

[implemented by TensorFlow] 

**Goal:**  
Using S&amp;P500 historical data from yahoo finance to predict S&amp;P500 price for the next day  

**Raw Data:**   
S&amp;P500 historical data from yahoo finance (1980.01.02 - 2017.11.17)

**ML Algorithms (implemented by TensorFlow):**  
Feedforward Artificial Neural Network.

**Plotting:**  
matplotlib  

**Envs:**  
Anaconda and Python 3.5  

**TensorFlow Install:**  
https://www.tensorflow.org/install/  

**Packages:**   
conda install numpy pandas matplotlib  
conda install spyder  

**Run Steps:**  
% git clone https://github.com/jasonx1011/stock-tensorflow.git  
% python stock.py  
or
using **spyder** to run stock.py (Recommended)  

**Sample Outputs:**  
   * Plotting Outputs:  
![sample_plot_1](./assets/result_training_set_full.png)  
![sample_plot_2](./assets/result_training_set_last_60_days.png)  
![sample_plot_3](./assets/result_validation_set_full.png)  
![sample_plot_4](./assets/result_validation_set_last_60_days.png)  

**Program Flow:**  
   * import raw data (pandas)  
   * preprocess data (pandas & numpy)  
   * no normalization & shuffle for training  
      * the idea is to preserve the time series information or pattern  
   * build mlp_net (TensorFlow)  
   * build the graph and train (TensorFlow & TensorBoard)  
   * run grid search for fine tuning hyperparameters  
   * compare results with Naive Model  
      * Naive Model: simply take mean() of the features as the prediction value  
   * output tensorboard meta-data or txt logs or images  
   * explore the data of multple runs by using TensorBoard  
