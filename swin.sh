#!/usr/local/bin/nosh

#$ -S /usr/local/bin/nosh
#$ -cwd

python3 main.py --model Swin --epochs 15 --batchsize 64
