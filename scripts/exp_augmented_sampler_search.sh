nohup python accuracy_globalsearch.py --dataset texas --embedding_tensor_name aug001_texas >> ./logs/exp_aug_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cornell --embedding_tensor_name aug001_cornell >> ./logs/exp_aug_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset wisconsin --embedding_tensor_name aug001_wisconsin >> ./logs/exp_aug_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cora --embedding_tensor_name aug001_cora >> ./logs/exp_aug_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset citeseer --embedding_tensor_name aug001_citeseer >> ./logs/exp_aug_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset photo --embedding_tensor_name aug001_photo >> ./logs/exp_aug_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset dblp --embedding_tensor_name aug001_dblp >> ./logs/exp_aug_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cs --embedding_tensor_name aug001_cs >> ./logs/exp_aug_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset physics --embedding_tensor_name aug001_physics >> ./logs/exp_aug_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset reddit  --embedding_tensor_name aug001_reddit >> ./logs/exp_aug_search.txt 2>&1 &