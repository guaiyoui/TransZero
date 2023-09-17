from ctc import ctc_cs_my
from ktruss import ktruss_cs
from ktruss_my import ktruss_cs_my
from kcore_my import kcore_cs_my
from kecc import kecc_cs
from kecc_my import kecc_cs_my
from utils import f1_score_calculation, load_query_n_gt
import argparse
import torch
from tqdm import tqdm

def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--dataset', type=str, default='cora', help='the name of the dataset')
    parser.add_argument('--method', type=str, default='kcore', help='the CS baselines')
    parser.add_argument('--path', type=str, default='../../dataset/', help='the path of the dataset')

    return parser.parse_args()


def index2binary(index, node_num):
    vec = [0 for i in range(node_num)]
    if index is None:
        return vec
    for i in range(len(index)):
        vec[index[i]] = 1
    return vec


if __name__ == "__main__":
    args = parse_args()
    edge_path = args.path + args.dataset + "/" + args.dataset + ".edges"

    node_num = {"cora":2708, "citeseer":3327}

    if args.method == "kcore":
        model = kcore_cs_my(edge_path)
    
    elif args.method == "ktruss":
        model = ktruss_cs(edge_path)

    elif args.method == "ctc":
        model = ctc_cs_my(edge_path)
    
    elif args.method == "ktruss_cs_my":
        model = ktruss_cs_my(edge_path)
    
    elif args.method == "kecc_cs":
        model = kecc_cs(edge_path)
    
    elif args.method == "kecc_cs_my":
        model = kecc_cs_my(edge_path)

    else:
        print("not valid method.")
        
    # load query and load ground truth
    query, gt = load_query_n_gt(args.path, args.dataset, node_num[args.dataset])

    prediction = []
    for i in tqdm(range(len(query))):
        component = model.query(query_set = query[i])
        component = index2binary(component, node_num[args.dataset])
        # print(component)
        prediction.append(component)
    
    f1score = f1_score_calculation(torch.Tensor(prediction), gt)
    print("The F1-score for: " + args.dataset+ ". by: "+ args.method + " is: " + str(f1score))

    
    
    


    

