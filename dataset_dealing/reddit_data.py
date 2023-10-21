import torch

data_list = torch.load("../reddit.pt")

labels = data_list[2]
torch.manual_seed(0)

print(labels, torch.min(labels), torch.max(labels))
num_class = torch.max(labels)-torch.min(labels)+1
print(num_class)
communities = [[i for i in range(labels.shape[0]) if labels[i]==j] for j in range(num_class)]

# print(communities, len(communities))

selected_queries = []
ground_truth = []

for i in range(150):
    num_node = torch.randint(1, 4, (1,)).item()
    selected_class = torch.randint(0, num_class, (1,)).item()
    print(num_node)
    selected_nodes = []
    for j in range(num_node):
        selected_node = torch.randint(0, len(communities[selected_class]), (1,)).item()
        selected_nodes.append(communities[selected_class][selected_node])
    selected_queries.append(selected_nodes)
    ground_truth.append(communities[selected_class])


query_file=open("reddit.query","w")
gt_file = open("reddit.gt", "w")

for i in range(len(selected_queries)):
    for j in range(len(selected_queries[i])):
        query_file.write(str(selected_queries[i][j]))
        query_file.write(" ")
    query_file.write("\n")
    for j  in range(len(ground_truth[i])):
        gt_file.write(str(ground_truth[i][j]))
        gt_file.write(" ")
    gt_file.write("\n")