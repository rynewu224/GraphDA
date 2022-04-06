# Graph Domain Adaptation: A Generative View
The official implementation of Graph Domain Adaptation: A Generative View. 
The model is a combination of Graph Neural Networks and some well-known DA frameworks, e.g, replacing the ResNet feature extractor with GCNs. 

## Requirements
* numpy==1.15.1
* scikit_learn==0.24.1
* torch==1.8.0

## Run the demo
```bash
python run_dann.py
```

## Data
The egonet dataset with 4 domains: OAG, Digg, Twitter and Weibo
* raw: https://github.com/xptree/DeepInf
* processed: https://pan.baidu.com/s/1BLxGa7785xuE17j_uVd2bA psw:9ins 

## Compared Models:
* DANN: https://www.jmlr.org/papers/volume17/15-239/15-239.pdf
* MDD: https://arxiv.org/abs/1904.05801
* DIVA: https://arxiv.org/abs/1905.10427
* DSR: https://arxiv.org/abs/2012.11807
