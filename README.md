# stock-tensorflow

[`TensorFlow` and `TensorBoard`] 

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
```  
conda install numpy pandas matplotlib  
conda install spyder  
```  

**Run Steps:**  
```  
git clone https://github.com/jasonx1011/stock-tensorflow.git  
python stock.py  
``` 
or
using `spyder` to run stock.py (Recommended)  

**Sample Outputs:**  
   * Plotting Outputs:  
![sample_plot_1](./assets/result_training_set_full.png)  
![sample_plot_2](./assets/result_training_set_last_60_days.png)  
![sample_plot_3](./assets/result_validation_set_full.png)  
![sample_plot_4](./assets/result_validation_set_last_60_days.png)  

**Results (may fluctuate):**  
Final total_loss:  
  
rmse_train= 3.951, lr_1E-03,2000,32,h_layers_16_32  
max/min difference: 3.209 to -13.417  
  
rmse_valid= 5.836, lr_1E-03,2000,32,h_layers_16_32  
max/min difference: 4.704 to -20.143  
  
Naive model:   
  
rmse_train= 4.243  
max/min difference: 33.170 to -46.265  
  
rmse_valid= 9.078  
max/min difference: 59.725 to -55.525  
  
**Program Flow:**  
   * import raw data (`pandas`)  
   * preprocess data (`pandas` & `numpy`)  
   * multiple features  
   * no normalization & shuffle for training  
      * the idea is to preserve the time series information or pattern  
   * build mlp_net (`TensorFlow`)  
   * build the graph and train (`TensorFlow` & `TensorBoard`)  
   * run grid search for fine tuning hyperparameters (learning rate, batch size, hidden layers, features ...etc)  
   * compare results with Naive Model  
      * Naive Model: simply take mean() of the features as the prediction value  
   * output tensorboard meta-data or txt logs or images  
   * explore the data of multple runs by using `TensorBoard`  
