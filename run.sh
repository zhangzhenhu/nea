#!/usr/bin/env bash
export MKL_THREADING_LAYER=GNU
THEANO_FLAGS="device=cpu,floatX=float32" ~/zhangzhenhu/env/bin/python train_nea.py \
	-tr data/AIlab/train.tsv \
	-tu data/fold_0/dev.tsv \
	-ts data/AIlab/test.tsv \
	-p 1 \
	--emb /home/work/zhangzhenhu/Semantic-Texual-Similarity-Toolkits/data/GoogleNews-vectors-negative300.txt \
	--epochs 20 \
	-o output
