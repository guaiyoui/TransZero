python link_pretrain.py --dataset pubmed --model_name pubmed --batch_size 19717 --epochs 100  --dropout 0.1 --hidden_dim 256 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0

python accuracy_globalsearch.py --dataset pubmed