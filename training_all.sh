
nohup  python link_pretrain.py --dataset texas --model_name texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/texas_training.txt 2>&1 &&

python link_pretrain.py --dataset cornell --model_name cornell --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/cornell_training.txt 2>&1 &&

python link_pretrain.py --dataset wisconsin --model_name wisconsin --batch_size 251 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/wisconsin_training.txt 2>&1 &&

python link_pretrain.py --dataset cora --model_name cora --batch_size 2708 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --device 1 >> ./logs/cora_training.txt 2>&1 &&

python link_pretrain.py --dataset citeseer --model_name citeseer --batch_size 3327  --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/citeseer_training.txt 2>&1 &&

python link_pretrain.py --dataset photo --model_name photo --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/photo_training.txt 2>&1 &&

python link_pretrain.py --dataset dblp --model_name dblp --batch_size 17716 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/dblp_training.txt 2>&1 &&

python link_pretrain.py --dataset cs --model_name cs --batch_size 18333 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 3 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/cs_training.txt 2>&1 &&

python link_pretrain.py --dataset physics --model_name physics --batch_size 4000 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/physics_training.txt 2>&1 &&

python link_pretrain.py --dataset reddit --model_name reddit --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 2 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/reddit_training.txt 2>&1 &
