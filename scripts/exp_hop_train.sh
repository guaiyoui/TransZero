nohup  python link_pretrain.py --dataset texas --model_name hop001_texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 1  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset texas --model_name hop002_texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 2  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset texas --model_name hop003_texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset texas --model_name hop004_texas   --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 4  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset texas --model_name hop006_texas   --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 6  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cornell --model_name hop001_cornell  --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 1  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cornell --model_name hop002_cornell   --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 2  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cornell --model_name hop003_cornell   --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cornell --model_name hop004_cornell   --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 4  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cornell --model_name hop006_cornell   --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 6  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset wisconsin --model_name hop001_wisconsin  --batch_size 251 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 1  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset wisconsin --model_name hop002_wisconsin   --batch_size 251 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 2  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset wisconsin --model_name hop003_wisconsin   --batch_size 251 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset wisconsin --model_name hop004_wisconsin   --batch_size 251 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 4  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset wisconsin --model_name hop006_wisconsin   --batch_size 251 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 6  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cora --model_name hop001_cora  --batch_size 2708 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 1  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cora --model_name hop002_cora   --batch_size 2708 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 2  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cora --model_name hop003_cora   --batch_size 2708 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cora --model_name hop004_cora   --batch_size 2708 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 4  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cora --model_name hop006_cora   --batch_size 2708 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 6  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset citeseer --model_name hop001_citeseer  --batch_size 3327  --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 1  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset citeseer --model_name hop002_citeseer   --batch_size 3327  --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 2  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset citeseer --model_name hop003_citeseer   --batch_size 3327  --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset citeseer --model_name hop004_citeseer   --batch_size 3327  --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 4  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset citeseer --model_name hop006_citeseer   --batch_size 3327  --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 6  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset photo --model_name hop001_photo  --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 1  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset photo --model_name hop002_photo   --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 2  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset photo --model_name hop003_photo   --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset photo --model_name hop004_photo   --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 4  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset photo --model_name hop006_photo   --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 6  --n_heads 8 --n_layers 1 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset dblp --model_name hop001_dblp  --batch_size 17716 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 1  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset dblp --model_name hop002_dblp   --batch_size 17716 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 2 --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset dblp --model_name hop003_dblp   --batch_size 17716 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset dblp --model_name hop004_dblp   --batch_size 17716 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 4  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset dblp --model_name hop006_dblp   --batch_size 17716 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 6  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cs --model_name hop001_cs  --batch_size 18333 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 1  --n_heads 8 --n_layers 3 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cs --model_name hop002_cs   --batch_size 18333 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 2  --n_heads 8 --n_layers 3 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset cs --model_name hop003_cs   --batch_size 18333 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 3 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset physics --model_name hop001_physics  --batch_size 4000 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 1  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset physics --model_name hop002_physics   --batch_size 4000 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 2  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset physics --model_name hop003_physics   --batch_size 4000 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset physics --model_name hop004_physics   --batch_size 4000 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 4  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset physics --model_name hop006_physics   --batch_size 4000 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 6  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset reddit --model_name hop001_reddit --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 1  --n_heads 8 --n_layers 2 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset reddit --model_name hop002_reddit --alpha 0.01  --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 2  --n_heads 8 --n_layers 2 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset reddit --model_name hop003_reddit --alpha 0.01  --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 2 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset reddit --model_name hop004_reddit --alpha 0.01  --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 4  --n_heads 8 --n_layers 2 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &&

python link_pretrain.py --dataset reddit --model_name hop006_reddit  --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 6  --n_heads 8 --n_layers 2 --pe_dim 10 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/exp_hop_training.txt 2>&1 &

