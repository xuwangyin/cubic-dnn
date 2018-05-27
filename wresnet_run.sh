#!/bin/bash

model=wresnet
mkdir $model
echo "/localtmp/yy8ms/anaconda3/bin/python wresnet_main.py  --save-dir=save_$model |& tee -a $model/log_$model_$1"
/localtmp/yy8ms/anaconda3/bin/python wresnet_main.py   --save-dir=save_$model |& tee -a $model/log_$model_$1