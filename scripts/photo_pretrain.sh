python link_pretrain.py --dataset photo --model_name photo --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0

python accuracy_globalsearch.py --dataset photo