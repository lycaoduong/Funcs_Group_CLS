# Funcs_Group_CLS
- Ohlabs FCG Classification
- Author: lycaoduong
- Email: lycaoduong@gmail.com

## Current Approach
- Transformer Encoder
- Transformer Decoder
- Multi-class Classification

## Environment Setting
```
conda env create -f environment.yml
```

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
--model [model name]: choosing model, Default: Fcg-test
--cktp [ckpt dir]: load pretrained weighted
--device [select device]: cuda or cpu
--input [input path]: input (npy) file path
```