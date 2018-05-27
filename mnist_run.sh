#!/bin/bash

model=mnist
echo "/localtmp/yy8ms/anaconda3/bin/python mnist_main.py  --save-dir=save_$model |& tee -a log_$model_$1"
/localtmp/yy8ms/anaconda3/bin/python mnist_main.py   --save-dir=save_$model |& tee -a log_$model_$1