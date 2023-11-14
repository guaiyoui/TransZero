nohup python accuracy_globalsearch.py --dataset texas --embedding_tensor_name epoch025_texas >> ./logs/exp_epoch_search_25.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cornell --embedding_tensor_name epoch025_cornell >> ./logs/exp_epoch_search_25.txt 2>&1 &&
python accuracy_globalsearch.py --dataset wisconsin --embedding_tensor_name epoch025_wisconsin >> ./logs/exp_epoch_search_25.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cora --embedding_tensor_name epoch025_cora >> ./logs/exp_epoch_search_25.txt 2>&1 &&
python accuracy_globalsearch.py --dataset citeseer --embedding_tensor_name epoch025_citeseer >> ./logs/exp_epoch_search_25.txt 2>&1 &&
python accuracy_globalsearch.py --dataset photo --embedding_tensor_name epoch025_photo >> ./logs/exp_epoch_search_25.txt 2>&1 &&
python accuracy_globalsearch.py --dataset dblp --embedding_tensor_name epoch025_dblp >> ./logs/exp_epoch_search_25.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cs --embedding_tensor_name epoch025_cs >> ./logs/exp_epoch_search_25.txt 2>&1 &&
python accuracy_globalsearch.py --dataset physics --embedding_tensor_name epoch025_physics >> ./logs/exp_epoch_search_25.txt 2>&1 &&
python accuracy_globalsearch.py --dataset reddit  --embedding_tensor_name epoch025_reddit >> ./logs/exp_epoch_search_25.txt 2>&1 &