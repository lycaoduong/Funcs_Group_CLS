# Funcs_Group_CLS
- Ohlabs FCG Classification
- Author: lycaoduong
- Email: lycaoduong@gmail.com

## Todo list
- Preprocessing dataset
- Evaluation code
- Reimplement paper's model
- Implement generative model

## Current Approach
- Transformer Encoder
- Transformer Decoder
- Multi-class Classification

## Environment Setting
```
conda env create -f environment.yml
```

## Pretrained Model
https://1drv.ms/u/s!AihbU7PyEgbOjgietVFW_squY8QO?e=S2aoqC

## For training
Run python script train.py with variable parser arguments:
```
--project [project name]: where the model and logging are stored, default location ../run/[project name]
--model [model name]: choosing model, Default: Fcg-B
--dataset [dataset name]: choosing dataset, Default: ohlabsFcg
--ckpt [ckpt dir]: load pretrained weighted, if you don't want to use pretrained weighted, ignore this argument
--device [select device]: cuda or cpu
--epochs [num epochs]: setting number of training epochs
--batch_size [num batch]: setting batch size
--lr [learning rate]: setting learning rate
--optimizer [optimizer]: adam or sgd
```

## For testing
Run python script predict.py with variable parser arguments:
```
--project [project name]: where the model and logging are stored, default location ../run/[project name]
--model [model name]: choosing model, Default: Fcg-B
--cktp [ckpt dir]: load pretrained weighted
--device [select device]: cuda or cpu
--input [input path]: input (npy) file path
```