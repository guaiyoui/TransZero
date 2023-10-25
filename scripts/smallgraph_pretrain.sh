python link_pretrain.py --dataset texas --model_name texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 

python accuracy_globalsearch.py --dataset texas

python link_pretrain.py --dataset cornell --model_name cornell --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 

python accuracy_globalsearch.py --dataset cornell

python link_pretrain.py --dataset wisconsin --model_name wisconsin --batch_size 251 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 

python accuracy_globalsearch.py --dataset wisconsin