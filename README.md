# GW model: 
model description...

#### Code Authors
Chen-Zhi Sua, ...


### Folders: 
* Original data: "./data/raw"
* Cleaned data: "./data/ftr"
* Training results: "./data/result"
* Predict results: "./data/pred"
* ML models: "./data/model"
* Plots: "./data/plot"
* Log: "./log"


### Environment Setting:
* python=3.7
* GPU version:??
* Install NVDIA GPU driver (for python=3.6 or 3.8):
```
pip install --user nvidia-pyindex
pip install --user nvidia-tensorflow[horovod]
```
* Install python packages:
```
pip install -r requirement.txt
```

### Modules:
* main.py: create random sampling training result.
* multi_exp.py: create group sampling training result.