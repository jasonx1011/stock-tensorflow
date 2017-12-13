# stock-tensorflow

[`TensorFlow` and `TensorBoard`] 

**Summary**  
Predicted the daily S&P500 index price using TensorFlow with multiple input features, validation error ~= 1%  
  
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

**Sample Plots:**  
   * Iutputs (before scaling):  
![sample_plot_inputs_1](./assets/inputs_training_set_last_60_days.png)  
![sample_plot_inputs_2](./assets/inputs_validation_set_last_60_days.png)  
![sample_plot_1](./assets/result_training_set_full.png)  
![sample_plot_2](./assets/result_training_set_last_60_days.png)  
   * Validation results (scaled):  
![sample_plot_3](./assets/result_validation_set_full.png)  
![sample_plot_4](./assets/result_validation_set_last_60_days.png)  
   * TensorBoard:  
![tb_plot_1](./assets/tb_graph.png)  
![tb_plot_2](./assets/tb_cost.png)  
  
**Results (may fluctuate):**  
---  
Final total_loss: 
  
rmse_train = 8.543, lr_3E-03,2000,128,h_layers_16_32  
error_train = 0.008, lr_3E-03,2000,128,h_layers_16_32  
max/min difference: 81.048 to -70.699  
  
rmse_valid = 18.863, lr_3E-03,2000,128,h_layers_16_32  
error_valid = 0.010, lr_3E-03,2000,128,h_layers_16_32  
max/min difference: 245.265 to -99.755  
---  
  
**Program Flow:**  
   * import raw data (`pandas`)  
   * preprocess data (`pandas` & `numpy`)  
   * multiple features  
   * build mlp_net (`TensorFlow`)  
   * build the graph and train (`TensorFlow` & `TensorBoard`)  
   * run grid search for fine tuning hyperparameters (learning rate, batch size, hidden layers, features ...etc)  
   * compare results with Avg. Model  
      * Avg. Model: take mean() of the features as the prediction value  
   * output tensorboard meta-data or txt logs or images  
   * explore the data of multple runs by using `TensorBoard`  
