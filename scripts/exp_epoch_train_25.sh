nohup  python link_pretrain.py --dataset texas --model_name  epoch025_texas  --batch_size 183 --epochs 25 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/exp_epoch_training_25.txt 2>&1 &&

python link_pretrain.py --dataset cornell --model_name  epoch025_cornell  --batch_size 183 --epochs 25 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/exp_epoch_training_25.txt 2>&1 &&

python link_pretrain.py --dataset wisconsin --model_name  epoch025_wisconsin  --batch_size 251 --epochs 25 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/exp_epoch_training_25.txt 2>&1 &&

python link_pretrain.py --dataset cora --model_name  epoch025_cora  --batch_size 2708 --epochs 25 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --device 1 >> ./logs/exp_epoch_training_25.txt 2>&1 &&

python link_pretrain.py --dataset citeseer --model_name  epoch025_citeseer  --batch_size 3327  --epochs 25 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/exp_epoch_training_25.txt 2>&1 &&

python link_pretrain.py --dataset photo --model_name  epoch025_photo  --batch_size 7650 --epochs 25 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/exp_epoch_training_25.txt 2>&1 &&

python link_pretrain.py --dataset dblp --model_name  epoch025_dblp  --batch_size 17716 --epochs 25 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/exp_epoch_training_25.txt 2>&1 &&

python link_pretrain.py --dataset cs --model_name  epoch025_cs  --batch_size 18333 --epochs 25 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 3 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/exp_epoch_training_25.txt 2>&1 &&

python link_pretrain.py --dataset physics --model_name  epoch025_physics  --batch_size 4000 --epochs 25 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/exp_epoch_training_25.txt 2>&1 &&

python link_pretrain.py --dataset reddit --model_name epoch025_reddit --alpha 0.01 --epochs 25 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 2 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 1 >> ./logs/exp_epoch_training_25.txt 2>&1 &

