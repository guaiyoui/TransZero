nohup  python accuracy_globalsearch.py --dataset texas --embedding_tensor_name abla_contra_texas >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset texas --embedding_tensor_name abla_generative_texas >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset texas --embedding_tensor_name abla_base_texas >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset texas --embedding_tensor_name abla_contra_texas >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset texas --embedding_tensor_name abla_generative_texas >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset texas --embedding_tensor_name abla_base_texas >> ./logs/exp_abalation_search.txt 2>&1 &&

python accuracy_globalsearch.py --dataset cornell --embedding_tensor_name abla_contra_cornell >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cornell --embedding_tensor_name abla_generative_cornell >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cornell --embedding_tensor_name abla_base_cornell >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset cornell --embedding_tensor_name abla_contra_cornell >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset cornell --embedding_tensor_name abla_generative_cornell >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset cornell --embedding_tensor_name abla_base_cornell >> ./logs/exp_abalation_search.txt 2>&1 &&

python accuracy_globalsearch.py --dataset wisconsin --embedding_tensor_name abla_contra_wisconsin >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset wisconsin --embedding_tensor_name abla_generative_wisconsin >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset wisconsin --embedding_tensor_name abla_base_wisconsin >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset wisconsin --embedding_tensor_name abla_contra_wisconsin >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset wisconsin --embedding_tensor_name abla_generative_wisconsin >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset wisconsin --embedding_tensor_name abla_base_wisconsin >> ./logs/exp_abalation_search.txt 2>&1 &&

python accuracy_globalsearch.py --dataset cora --embedding_tensor_name abla_contra_cora >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cora --embedding_tensor_name abla_generative_cora >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cora --embedding_tensor_name abla_base_cora  >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset cora --embedding_tensor_name abla_contra_cora >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset cora --embedding_tensor_name abla_generative_cora >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset cora --embedding_tensor_name abla_base_cora  >> ./logs/exp_abalation_search.txt 2>&1 &&

python accuracy_globalsearch.py --dataset citeseer --embedding_tensor_name abla_contra_citeseer  >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset citeseer --embedding_tensor_name abla_generative_citeseer >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset citeseer --embedding_tensor_name abla_base_citeseer >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset citeseer --embedding_tensor_name abla_contra_citeseer  >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset citeseer --embedding_tensor_name abla_generative_citeseer >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset citeseer --embedding_tensor_name abla_base_citeseer >> ./logs/exp_abalation_search.txt 2>&1 &&

python accuracy_globalsearch.py --dataset photo --embedding_tensor_name abla_contra_photo >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset photo --embedding_tensor_name abla_generative_photo >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset photo --embedding_tensor_name abla_base_photo >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset photo --embedding_tensor_name abla_contra_photo >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset photo --embedding_tensor_name abla_generative_photo >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset photo --embedding_tensor_name abla_base_photo >> ./logs/exp_abalation_search.txt 2>&1 &&

python accuracy_globalsearch.py --dataset dblp --embedding_tensor_name abla_contra_dblp >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset dblp --embedding_tensor_name abla_generative_dblp >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset dblp --embedding_tensor_name abla_base_dblp >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset dblp --embedding_tensor_name abla_contra_dblp >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset dblp --embedding_tensor_name abla_generative_dblp >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset dblp --embedding_tensor_name abla_base_dblp >> ./logs/exp_abalation_search.txt 2>&1 &&

python accuracy_globalsearch.py --dataset cs --embedding_tensor_name abla_contra_cs >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cs --embedding_tensor_name abla_generative_cs >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cs --embedding_tensor_name abla_base_cs >> ./logs/cs_traiexp_abalation_searchning.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cs --embedding_tensor_name abla_contra_cs >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cs --embedding_tensor_name abla_generative_cs >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset cs --embedding_tensor_name abla_base_cs >> ./logs/cs_traiexp_abalation_searchning.txt 2>&1 &&

python accuracy_globalsearch.py --dataset physics --embedding_tensor_name abla_contra_physics >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset physics --embedding_tensor_name abla_generative_physics >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset physics --embedding_tensor_name abla_base_physics >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset physics --embedding_tensor_name abla_contra_physics >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset physics --embedding_tensor_name abla_generative_physics >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset physics --embedding_tensor_name abla_base_physics >> ./logs/exp_abalation_search.txt 2>&1 &&

python accuracy_globalsearch.py --dataset reddit --embedding_tensor_name abla_contra_reddit >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset reddit --embedding_tensor_name abla_generative_reddit >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_globalsearch.py --dataset reddit --embedding_tensor_name abla_base_reddit >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset reddit --embedding_tensor_name abla_contra_reddit >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset reddit --embedding_tensor_name abla_generative_reddit >> ./logs/exp_abalation_search.txt 2>&1 &&
python accuracy_localsearch.py --dataset reddit --embedding_tensor_name abla_base_reddit >> ./logs/exp_abalation_search.txt 2>&1 &
