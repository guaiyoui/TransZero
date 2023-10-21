python link_pretrain.py --dataset reddit --epochs 100 --batch_size 10000 --dropout 0.1 --hidden_dim 512 --hops 10  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 

nohup python3 -u link_pretrain.py --dataset reddit --model_name reddit --epochs 100 --batch_size 10000 --dropout 0.1 --hidden_dim 512 --hops 10  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 2 >> ./logs/reddit_1w.txt 2>&1 &

nohup python3 -u link_pretrain.py --dataset reddit --model_name reddit_2w --epochs 100 --batch_size 20000 --dropout 0.1 --hidden_dim 512 --hops 10  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 3 >> ./logs/reddit_2w.txt 2>&1 &

nohup python3 -u link_pretrain.py --dataset reddit --model_name reddit_2k --epochs 100 --batch_size 2000 --dropout 0.1 --hidden_dim 512 --hops 10  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 4 >> ./logs/reddit_2k.txt 2>&1 &

nohup python3 -u link_pretrain.py --dataset reddit --model_name reddit_4k --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 10  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 4 >> ./logs/reddit_4k.txt 2>&1 &