# ??? model: 

#### Code Authors
Chen-Zhi Sua, ...


### Folders: 
* Original data: "./data/raw"
* Cleaned data: "./data/ftr"
* Training results: "./data/result"
* Predict results: "./data/pred"
* ML models: "./data/model"
* Plots: "./data/plot"


### Environment Setting:
python=3.8
GPU version:
Install NVDIA GPU driver (for python=3.6 or 3.8):
pip install --user nvidia-pyindex
pip install --user nvidia-tensorflow[horovod]
Install python packages:
pip install -r requirement.txt

### Modules:
Flow: Data_Cleaner --> Create_Axon_Feature --> Prepared_Axon --> Classify_Axon  

* Data_Cleaner::load_data(): turn swc skeletons into level trees.

* Create_Axon_Feature::load_data(): create Soma-features and Local-features.

* Data_Augmentation::create_axon_featrue(): create features for fake neurons.

* Prepared_Axon::load_data(): prepare input data.
  
* Classify_Axon::
  * classify_data(): train and test over the data.
  
  * predict(): predict new data.
 
  * evaluate(): evaluate the trained model.
  ![alt text](img/evaluate.png)
  
  * plot_result():
  ![alt text](img/plot_result.png)
  
  * correct_distribution(): x: the accuracy of a nueron, y: the number of neurons under certain accuracy. Title: the name of the set of neurons (total neuron number).
  ![alt text](img/correct_distribution.png)
  
  * analyze_feature(): display the axon/dendrite distribution by a certain feature. Title: the name of the set of neurons (ftr importance in the model) (total node number).
  ![alt text](img/analyze_feature.png)
    
    
* Plot_Tree::plot_tree(): plot level trees.
![alt text](img/plot_tree.png)

