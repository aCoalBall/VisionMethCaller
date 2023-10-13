#!/usr/local/bin/bash

#$ -S /usr/local/bin/bash
#$ -cwd

cd /home/coalball/projects/methBert2/toys_1m 

python3 main.py --model ViT --epochs 15 --batchsize 32
