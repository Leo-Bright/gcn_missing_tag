import numpy as np
import pickle as pkl

from collections import defaultdict
from gcn.utils import parse_index_file

import random as rd
import scipy.sparse as sp
import json


def get_ndarray(items, dim=9):

    if len(items) == 1 and items != "-1":
        v = np.zeros(dim)
        v[items] = 1
        return v
    elif len(items) == 1 and items == "-1":
        return None
    else:
        map(float, items)
        v = np.array(items)
        return v


# supports emb file and tag file
def trans_input_file_to_ndarray(input):

    support_suffix = ["emb", "embedding", "embeddings", "tag"]

    if input.rsplit('.',1 )[-1] not in support_suffix:
        raise BaseException("Only support emb and tag file.")

    output_ndarray_dic = {}

    with open(input) as f:
        for line in f:
            node_others = line.strip().split(' ')
            node_id = node_others[0]
            others = node_others[1:]
            oth_array = get_ndarray(others, dim=9)
            if len(oth_array) > 0:
                output_ndarray_dic[node_id] = oth_array
    return output_ndarray_dic


def load_edgelist(file_, idx_dict, undirected=True):
    G = defaultdict(list)
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = idx_dict[x]
            y = idx_dict[y]
            G[x].append(y)
            if undirected:
                G[y].append(x)

    for k in G.keys():
        G[k] = list(sorted(set(G[k])))

    for x in G:
        if x in G[x]:
            G[x].remove(x)

    return G


def split_labeled_instance(all_labeled_samples_path, train_size, valid_size):

    with open(all_labeled_samples_path, "rb") as f:
        all_labeled_samples = pkl.load(f)

    rd.shuffle(all_labeled_samples)
    rd.shuffle(all_labeled_samples)
    rd.shuffle(all_labeled_samples)

    size = len(all_labeled_samples)

    x = all_labeled_samples[:train_size]

    valid = all_labeled_samples[train_size: train_size + valid_size]

    test = all_labeled_samples[train_size + valid_size:]

    print(len(test))

    with open("sanfrancisco/ind.sanfrancisco.x.index", "w+") as f:
        for item in x:
            f.write(str(item) + '\n')

    with open("sanfrancisco/ind.sanfrancisco.testx.index", "w+") as f:
        for item in test:
            f.write(str(item) + '\n')

    with open("sanfrancisco/ind.sanfrancisco.validx.index", "w+") as f:
        for item in valid:
            f.write(str(item) + '\n')

    return len(test)


def get_x_y_file(input_idx_file, node2tag, node2emb, idx2node, output=["x", "y"]):

    x = []

    y = []

    red = set()

    with open(input_idx_file) as f:
        for l in f:
            node_id = idx2node[int(l.strip())]
            if node_id not in node2emb:
                red.add(int(l.strip()))
                continue
            features = node2emb[node_id]
            # label = node2tag[node_id]
            label = [1, 0] if node_id in node2tag else [0, 1]

            new_features = []
            for i in range(len(features)):
                new_features.append(float(features[i]))

            new_label = label

            x.append(new_features)
            y.append(np.array(new_label))

    X = sp.csr_matrix(x)

    Y = np.array(y)

    with open("sanfrancisco/ind.sanfrancisco." + output[0], "wb") as f:
        pkl.dump(X, f)

    with open("sanfrancisco/ind.sanfrancisco." + output[1], "wb") as f:
        pkl.dump(Y, f)

    return red


# the samples that have not label
def get_other_x_y_file(idx_paths, node2emb, node2idx, network):

    other_idx = []

    x = []

    y = []

    idx_had = set()
    for path in idx_paths:
        with open(path, "r") as f:
            for l in f:
                idx_had.add(int(l.strip()))

    for node_id in node2emb:
        idx = node2idx[node_id]
        if idx in idx_had:
            continue
        else:
            idx_had.add(idx)
            other_idx.append(idx)
            features = node2emb[node_id]
            label = np.zeros(2)

            new_features = []
            for i in range(len(features)):
                new_features.append(float(features[i]))

            new_label = label
            x.append(new_features)
            y.append(np.array(new_label))

    X = sp.csr_matrix(x)

    Y = np.array(y)

    with open("sanfrancisco/ind.sanfrancisco.otherx", "wb") as f:
        pkl.dump(X, f)

    with open("sanfrancisco/ind.sanfrancisco.othery", "wb") as f:
        pkl.dump(Y, f)

    with open("sanfrancisco/ind.sanfrancisco.otherx.index", "w") as f:
        for i in other_idx:
            f.write(str(i) + '\n')

    network_idxs = set()

    for idx in network:
        network_idxs.add(idx)

    red = network_idxs - idx_had

    return red


def get_all_x_y_file():

    names = ['x', 'y', 'validx', 'validy', 'otherx', 'othery']
    objects = []
    for i in range(len(names)):
        with open("sanfrancisco/ind.sanfrancisco.{}".format(names[i]), 'rb') as f:
            objects.append(pkl.load(f))

    x, y, validx, validy, otherx, othery = tuple(objects)

    X = sp.vstack((x, validx, otherx))

    Y = np.vstack((y, validy, othery))

    with open("sanfrancisco/ind.sanfrancisco.allx", "wb") as f:
        pkl.dump(X, f)

    with open("sanfrancisco/ind.sanfrancisco.ally", "wb") as f:
        pkl.dump(Y, f)


def generate_global_idx(node2emb, idx2node, output_path):

    idx_paths = ["sanfrancisco/ind.sanfrancisco.x.index",
                 "sanfrancisco/ind.sanfrancisco.validx.index",
                 "sanfrancisco/ind.sanfrancisco.otherx.index",
                 "sanfrancisco/ind.sanfrancisco.testx.index"]

    test_idx_reorder = parse_index_file("sanfrancisco/ind.{}.test.index".format("sanfrancisco"))
    test_idx_range = np.sort(test_idx_reorder)

    feature_idx = []
    for i in range(len(idx_paths)):
        with open(idx_paths[i]) as f:
            for l in f:
                f_idx = int(l.strip())
                if idx2node[f_idx] not in node2emb:
                    continue
                feature_idx.append(int(l.strip()))

    print(len(feature_idx))

    features = np.array(feature_idx)

    features[test_idx_reorder] = features[test_idx_range]

    with open(output_path, "wb") as f:
        pkl.dump(features, f)


def remove_redundant_node(road_network, redundant_idx):

    for idx in redundant_idx:
        del road_network[idx]

    for k in road_network:
        nodes = road_network[k]
        road_network[k] = list(set(nodes) - redundant_idx)

    with open(graph_file_path, "wb") as graph_file:
        pkl.dump(road_network, graph_file)


def generate_network_graph(network_file_path, graph_file_path, node_idx_dict):

    graph = {}

    with open(network_file_path) as f:
        for line in f:
            ids = line.strip().split(' ')
            start = node_idx_dict[ids[0]]
            end = node_idx_dict[ids[1]]

            if start not in graph:
                graph[start] = []
            graph[start].append(end)

            if end not in graph:
                graph[end] = []
            graph[end].append(start)

    with open(graph_file_path + '.json', 'w+') as f:
        f.write(json.dumps(graph))

    with open(graph_file_path, 'wb') as f:
        pkl.dump(graph, f)


def gen_all_labeled_pkl(node2tag, node2idx, all_labeled_samples_path):

    all_labeled_samples_idx = []

    count_labeled = 0
    count_unlabeled = 0
    for key in node2idx:
        if key in node2tag:
            all_labeled_samples_idx.append(node2idx[key])
            count_labeled += 1
        elif count_unlabeled < count_labeled + 1:
            all_labeled_samples_idx.append(node2idx[key])
            count_unlabeled += 1

    print(len(all_labeled_samples_idx))

    with open(all_labeled_samples_path, 'wb') as f:
        pkl.dump(all_labeled_samples_idx, f)


def gen_node_idx_pkl_path(node2emb, node2idx_pkl_path, idx2node_pkl_path):

    node2idx = {}
    idx2node = {}

    count = 0
    for key in node2emb:
        node2idx[key] = count
        idx2node[count] = key
        count += 1

    with open(node2idx_pkl_path, 'wb') as f:
        pkl.dump(node2idx, f)

    with open(idx2node_pkl_path, 'wb') as f:
        pkl.dump(idx2node, f)


def gen_test_index_file(samples_num, output_path):

    test_idxs = list(range(58404 - samples_num, 58404))
    rd.shuffle(test_idxs)

    with open(output_path, 'w+') as f:
        for idx in test_idxs:
            f.write(str(idx) + '\n')


if __name__ == '__main__':

    with open("sanfrancisco/sf_node_idx_dict.pkl", "rb") as f:
        node_idx_dict = pkl.load(f)

    with open("sanfrancisco/sf_idx_node_dict.pkl", "rb") as f:
        idx_node_dict = pkl.load(f)

    with open("sanfrancisco/osm_data/nodes_turning_circle.json") as f:
        node_tag_dict = json.loads(f.readline())

    node_emb_dict = trans_input_file_to_ndarray('sanfrancisco/embeddings/sanfrancisco_raw_feature_none.embeddings')

    graph_file_path = "sanfrancisco/ind.sanfrancisco.graph"
    with open(graph_file_path, "rb") as f:
        network = pkl.load(f)

    # with open("sanfrancisco/ind.sanfrancisco.graph.json", "w+") as f:
    #     f.write(json.dumps(network))

    x_index = ['sanfrancisco/ind.sanfrancisco.x.index', 'x', 'y']
    test_x_index = ['sanfrancisco/ind.sanfrancisco.testx.index', 'tx', 'ty']
    valid_x_index = ['sanfrancisco/ind.sanfrancisco.validx.index', 'validx', 'validy']
    indexes_to_gen = [x_index, test_x_index, valid_x_index]

    # step1: generate x,testx,validx,y,testy,validy file
    # for index in indexes_to_gen:
    #     red_idx = get_x_y_file(index[0], node_tag_dict, node_emb_dict, idx_node_dict, output=index[1:])
    #     if len(red_idx) > 0:
    #         print('have redundat idx: ', len(red_idx))
    #         remove_redundant_node(network, red_idx, graph_file_path)

    # step2: generate otherx, othery file
    # gcn_emb_idx_path = 'sanfrancisco/embeddings/sf_gcn_raw_feature_none_16d_target_is_turning_circle.embedding.idx.pkl'
    # idx_paths = [x_index[0], test_x_index[0], valid_x_index[0]]
    # red_idx = get_other_x_y_file(idx_paths, node_emb_dict, node_idx_dict, network)
    # if len(red_idx) > 0:
    #     print('have redundat idx: ', len(red_idx))
    #     remove_redundant_node(network, red_idx, graph_file_path)
    # generate_global_idx(node_emb_dict, idx_node_dict, gcn_emb_idx_path)

    # step3: generate allx, ally files that use in train
    # get_all_x_y_file()

    ########################################
    # under is not main process
    ########################################

    # step0-0: generate the init indexes file to x, testx, validx
    # node_idx_pkl_path = 'sanfrancisco/sf_node_idx_dict.pkl'
    # idx_node_pkl_path = 'sanfrancisco/sf_idx_node_dict.pkl'
    # gen_node_idx_pkl_path(node_emb_dict, node_idx_pkl_path, idx_node_pkl_path)

    # step0-1: generate the road network graph
    # generate_network_graph('sanfrancisco/osm_data/sf_roadnetwork', 'sanfrancisco/ind.sanfrancisco.graph', node_idx_dict)

    # step0-2: generate idx pkl file of the all labeled samples that use to train/test/valid
    # all_labeled_pkl_path = 'sanfrancisco/ind.sanfrancisco.all.labeled.pkl'
    # gen_all_labeled_pkl(node_tag_dict, node_idx_dict, all_labeled_pkl_path)

    # step0-3: generate idx file of the all labeled samples that use to test
    # all_labeled_pkl_path = 'sanfrancisco/ind.sanfrancisco.all.labeled.pkl'
    # samples_size = split_labeled_instance(all_labeled_pkl_path, 1700, 200)
    # test_index_file_path = 'sanfrancisco/ind.sanfrancisco.test.index'
    # gen_test_index_file(samples_size, test_index_file_path)

    print("1")







