#!/bin/bash

model=resnet
echo "/localtmp/yy8ms/anaconda3/bin/python resnet_main.py  --save-dir=save_$model |& tee -a log_$model_$1"
/localtmp/yy8ms/anaconda3/bin/python resnet_main.py   --save-dir=save_$model |& tee -a log_$model_$1