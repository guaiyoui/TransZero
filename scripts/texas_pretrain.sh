python link_pretrain.py --dataset texas --model_name texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 
# python reconstruct_pretrain.py --dataset texas --model_name texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 

python accuracy_globalsearch.py --dataset texas