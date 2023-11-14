nohup python accuracy_localsearch.py --dataset texas >> ./logs/test_all_local.txt 2>&1 &&
python accuracy_localsearch.py --dataset cornell >> ./logs/test_all_local.txt 2>&1 &&
python accuracy_localsearch.py --dataset wisconsin >> ./logs/test_all_local.txt 2>&1 &&
python accuracy_localsearch.py --dataset cora >> ./logs/test_all_local.txt 2>&1 &&
python accuracy_localsearch.py --dataset citeseer >> ./logs/test_all_local.txt 2>&1 &&
python accuracy_localsearch.py --dataset photo >> ./logs/test_all_local.txt 2>&1 &&
python accuracy_localsearch.py --dataset dblp >> ./logs/test_all_local.txt 2>&1 &&
python accuracy_localsearch.py --dataset cs >> ./logs/test_all_local.txt 2>&1 &&
python accuracy_localsearch.py --dataset physics >> ./logs/test_all_local.txt 2>&1 &&
python accuracy_localsearch.py --dataset reddit  >> ./logs/test_all_local.txt 2>&1 &