#!/usr/local/bin/nosh

#$ -S /usr/local/bin/nosh
#$ -cwd

source ~/venvs/methbert2_venv/bin/activate
python3 main.py --model tinyViT --epochs 15 --batchsize 64
