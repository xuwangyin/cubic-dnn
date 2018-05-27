#!/bin/bash

model=dpn
mkdir $model
echo "/localtmp/yy8ms/anaconda3/bin/python dpn_main.py  --save-dir=save_$model |& tee -a $model/log_$model_$1"
/localtmp/yy8ms/anaconda3/bin/python dpn_main.py   --save-dir=save_$model |& tee -a $model/log_$model_$1