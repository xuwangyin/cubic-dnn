#!/bin/bash

model=dn
mkdir $model
echo "/localtmp/yy8ms/anaconda3/bin/python dn_main.py  --save-dir=save_$model |& tee -a $model/log_$model_$1"
/localtmp/yy8ms/anaconda3/bin/python dn_main.py   --save-dir=save_$model |& tee -a $model/log_$model_$1