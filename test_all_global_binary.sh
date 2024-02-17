nohup python accuracy_globalsearch_binary.py --dataset texas >> ./logs/test_all_global_binary.txt 2>&1 &&
python accuracy_globalsearch_binary.py --dataset cornell >> ./logs/test_all_global_binary.txt 2>&1 &&
python accuracy_globalsearch_binary.py --dataset wisconsin >> ./logs/test_all_global_binary.txt 2>&1 &&
python accuracy_globalsearch_binary.py --dataset cora >> ./logs/test_all_global_binary.txt 2>&1 &&
python accuracy_globalsearch_binary.py --dataset citeseer >> ./logs/test_all_global_binary.txt 2>&1 &&
python accuracy_globalsearch_binary.py --dataset photo >> ./logs/test_all_global_binary.txt 2>&1 &&
python accuracy_globalsearch_binary.py --dataset dblp >> ./logs/test_all_global_binary.txt 2>&1 &&
python accuracy_globalsearch_binary.py --dataset cs >> ./logs/test_all_global_binary.txt 2>&1 &&
python accuracy_globalsearch_binary.py --dataset physics >> ./logs/test_all_global_binary.txt 2>&1 &&
python accuracy_globalsearch_binary.py --dataset reddit  >> ./logs/test_all_global_binary.txt 2>&1 &