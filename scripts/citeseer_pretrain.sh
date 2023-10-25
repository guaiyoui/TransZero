python link_pretrain.py --dataset citeseer --model_name citeseer --batch_size 3327 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --epochs 100 
# python reconstruct_pretrain.py --dataset citeseer --model_name citeseer --batch_size 3327 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --epochs 100 

python accuracy_globalsearch.py --dataset citeseer