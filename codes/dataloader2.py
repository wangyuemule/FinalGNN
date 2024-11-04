import torch
#from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
# import numpy as np
import os
import pandas as pd
# from collections import Counter
# import json
import gc

class Dataset2(InMemoryDataset):
    def __init__(self, root, dataset, pred_edges=1, sep=' ', sufix='', transform=None, pre_transform=None):

        self.sep = sep
        self.sufix = sufix
        self.path = root
        self.dataset = dataset
        self.pred_edges = pred_edges
        self.store_backup = True

        super(Dataset2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistical_info = torch.load(self.processed_paths[1])
        self.feature_num = self.statistical_info['feature_num']
        self.data_num = self.statistical_info['data_num']
        self.edge_num = self.statistical_info['edge_num']
        self.field_num = self.statistical_info['field_num']

    @property
    def raw_file_names(self):
        '''
        return ['{}{}/{}.data'.format(self.path, self.dataset, self.dataset), \
                '{}{}/{}.edge'.format(self.path, self.dataset, self.dataset)]
        '''
        return ['{}{}/{}.data'.format(self.path, self.dataset, self.dataset)]

    @property
    def processed_file_names(self):
        if not self.pred_edges:
            return ['{}/{}.dataset'.format(self.dataset, self.dataset), \
                    '{}/{}.info'.format(self.dataset, self.dataset)]
        else:
            return ['{}/{}.dataset'.format(self.dataset, self.dataset), \
                    '{}/{}.info'.format(self.dataset, self.dataset)]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def data_2_graphs(self, ratings_df, dataset='train'):
        graphs = []
        processed_graphs = 0
        num_graphs = ratings_df.shape[0]
        one_per = int(num_graphs/1000)
        percent = 0.0
        for i in range(len(ratings_df)):
            if processed_graphs % one_per == 0:
                print('\r Processing [{}]: {}%, {}/{}'.format(dataset, percent/10.0, processed_graphs, num_graphs), end='')
                percent += 1
            processed_graphs += 1
            line = ratings_df.iloc[i]
            feature_list = [int(line[i]) for i in range(len(line))[1:]]
            rating = int(line[0])

            graph = self.construct_graph(feature_list, rating)
            graphs.append(graph)

        return graphs

    def read_data(self):
        ratings_df = pd.read_csv(self.datafile, sep=self.sep, header=None)
        train_df, test_df = train_test_split(ratings_df, test_size=0.15, random_state=2024, stratify=ratings_df[0])
        train_df, valid_df = train_test_split(train_df, test_size=15 / 85, random_state=2024, stratify=train_df[0])

        stat_info = {}
        tempt = ratings_df.iloc[:, 1:]
        stat_info['feature_num'] = max(tempt.max(axis=1)) - min(tempt.min(axis=1)) + 1
        stat_info['field_num'] = tempt.shape[1]
        stat_info['edge_num'] = tempt.shape[1] ** 2  
        del ratings_df, tempt
        gc.collect()

        if self.store_backup:
            backup_path = f"{self.path}{self.dataset}/split_data_backup/"
            if not os.path.exists(backup_path):
                os.mkdir(backup_path)

            train_df.to_csv(f'{backup_path}train_data.csv', header=False, index=False)
            valid_df.to_csv(f'{backup_path}valid_data.csv', header=False, index=False)
            test_df.to_csv(f'{backup_path}test_data.csv', header=False, index=False)

        train_graphs = self.data_2_graphs(train_df, dataset='train')
        del train_df
        gc.collect()
        valid_graphs = self.data_2_graphs(valid_df, dataset='valid')
        del valid_df
        gc.collect()
        test_graphs = self.data_2_graphs(test_df, dataset='test')
        del test_df
        gc.collect()
        #print(len(train_graphs), len(valid_graphs), len(test_graphs))
        graphs = train_graphs + valid_graphs + test_graphs
        stat_info['data_num'] = len(graphs)
        return graphs, stat_info


    def construct_full_edge_list(self, nodes):
        num_node = len(nodes)
        edge_list = [[], []]  
        receiver_sender_list = []
        for i in range(num_node): 
            for j in range(num_node):
                edge_list[0].append(i)
                edge_list[1].append(j)
                receiver_sender_list.append([nodes[i], nodes[j]])

        return edge_list, receiver_sender_list

    def construct_graph(self, node_list, label):
        edge_l, sr_l = self.construct_full_edge_list(node_list)
        x = torch.LongTensor(node_list).unsqueeze(1)
        edge_index = torch.LongTensor(edge_l)
        y = torch.FloatTensor([label])
        sr = torch.LongTensor(sr_l)  

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=sr, y=y)  
        return graph_data

    def process(self):
        self.datafile = self.raw_file_names[0] 
        graphs, statistical_info = self.read_data()

        if not os.path.exists(f"{self.path}/processed/{self.dataset+self.sufix}"):
            os.mkdir(f"{self.path}/processed/{self.dataset+self.sufix}")

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

        torch.save(statistical_info, self.processed_paths[1])

    def node_N(self):
        return self.feature_num

    def data_N(self):
        return self.data_num

    def edge_N(self):
        return self.edge_num

    def field_N(self):
        return self.field_num


