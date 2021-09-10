#!/bin/bash
source activate py36-curriculas
python3 metric_learning.py --batch 32 --dim_out 512
python3 metric_learning.py --batch 64 --dim_out 512
python3 metric_learning.py --batch 128 --dim_out 512
python3 metric_learning.py --batch 32 --dim_out 256
python3 metric_learning.py --batch 64 --dim_out 256
python3 metric_learning.py --batch 128 --dim_out 256
python3 metric_learning.py --batch 32 --dim_out 128
python3 metric_learning.py --batch 64 --dim_out 128
python3 metric_learning.py --batch 128 --dim_out 128
