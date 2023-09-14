python link_pretrain.py --dataset texas --model_name texas --batch_size 187 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --epochs 100 

python maxtopk_accuracy.py --dataset texas
python kmeans_accuracy.py --dataset texas --num_communities 5