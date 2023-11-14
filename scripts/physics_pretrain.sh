# 0.3291
python link_pretrain.py --dataset physics --model_name physics --batch_size 4000 --alpha 0.01 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --epochs 100 --device 1

# 0.5423
# python link_pretrain.py --dataset physics --model_name physics --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --epochs 100 --device 1

python accuracy_globalsearch.py --dataset physics