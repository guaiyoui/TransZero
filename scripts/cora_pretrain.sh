# python personalize_pretrain.py --dataset cora --batch_size 2000 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 
# python link_pretrain.py --dataset cora --batch_size 2708 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --epochs 100
# python gcn_pretrain.py --dataset cora --batch_size 2708 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --epochs 100
# python kmeanslink_pretrain.py --dataset cora --batch_size 2708 --group_epoch_gap 80 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --epochs 100
# python pq_pretrain.py --dataset cora --batch_size 2708 --group_epoch_gap 60 --dropout 0.1 --hidden_dim 512 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --epochs 100
# python maxtopk_accuracy.py
# python kmeans_accuracy.py
python accuracy_maxtopk.py
python accuracy_kmeans.py

