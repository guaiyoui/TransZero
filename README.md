code for the paper "Efficient Unsupervised Community Search with Pre-trained Graph Transformer"

[![Awesome](https://awesome.re/badge.svg)](https://github.com/guaiyoui/graph-analytics-starter-pack) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)



### Fast Start

```
1: python link_pretrain.py
2: python accuracy_globalsearch.py
```

### Train all datasets
```
bash ./training_all.sh
```

### Test all datasets
```
bash ./test_all_global.sh >> ./logs/test_all_global.txt 2>&1 &
bash ./test_all_local.sh >> ./logs/test_all_local.txt 2>&1 &
```


### Folder Structure

    .
    ├── dataset                     # make a new folder by "mkdir dataset"
    ├── dataset_dealing             # the scripts to download datasets and deal datasets automatically
    ├── logs                        # the running logs
    ├── model                       # the saved model
    ├── pretrain_result             # the pretrained latent representation
    ├── scripts                     # the scripts to run the model and the experiments
    ├── accuracy_globalsearch.py    # the MESG solver-Global_Binary_Search
    ├── accuracy_localsearch.py     # the MESG solver-Local_Search
    ├── data_loader.py              # data loader
    ├── early_stop.py               # early stop module to alleviate overfitting
    ├── layer.py                    # the layer in the network
    ├── link_pretrain.py            # the overall entrance for the model
    ├── layer.py                    # the layer in the network
    ├── lr.py                       # the learning rate module
    ├── model.py                    # the model definition
    ├── utils.py                    # the utils used
    ├── test_all_global.sh          # test the performance of all datasets by global binary search
    ├── test_all_local.sh           # test the performance of all datasets by local search
    ├── training_all.sh             # the script to train all the models
    └── README.md

