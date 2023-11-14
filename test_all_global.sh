nohup python accuracy_globalsearch.py --dataset texas >> ./logs/test_all_global.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cornell >> ./logs/test_all_global.txt 2>&1 &&
python accuracy_globalsearch.py --dataset wisconsin >> ./logs/test_all_global.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cora >> ./logs/test_all_global.txt 2>&1 &&
python accuracy_globalsearch.py --dataset citeseer >> ./logs/test_all_global.txt 2>&1 &&
python accuracy_globalsearch.py --dataset photo >> ./logs/test_all_global.txt 2>&1 &&
python accuracy_globalsearch.py --dataset dblp >> ./logs/test_all_global.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cs >> ./logs/test_all_global.txt 2>&1 &&
python accuracy_globalsearch.py --dataset physics >> ./logs/test_all_global.txt 2>&1 &&
python accuracy_globalsearch.py --dataset reddit  >> ./logs/test_all_global.txt 2>&1 &