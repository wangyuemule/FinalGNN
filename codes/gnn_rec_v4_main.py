from dataloader2 import Dataset2
import argparse
from torch_geometric.loader import DataLoader
from gnn_rec_v4_train import train
import torch
import random
import numpy as np
import colorsys
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='frappe', help='which dataset to use')
parser.add_argument('--embedding_dim', type=int, default=64, help='dimension of entity and relation embeddings')
parser.add_argument('--field_dim', type=int, default=64, help='dimension of field embeddings')
parser.add_argument('--hidden_dim', type=int, default=32, help='neural hidden layer')
parser.add_argument('--l0_weight', type=float, default=0.001, help='weight of the l0 regularization term')
parser.add_argument('--au_weight', type=float, default=0.5, help='weight of the auxiliary term')
parser.add_argument('--l2_weight', type=float, default=0.00001, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--alpha', type=float, default=0.4, help='layer weight')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--l0_para', nargs='?', default='[0.66, -0.1, 1.1]',
                        help="l0 parameters, which are beta (temprature), zeta (interval_min) and gama (interval_max).")
parser.add_argument('--head_num', type=int, default=2, help='The number of attention head')
parser.add_argument('--highest_num', type=int, default=2, help='The number of interaction time')
parser.add_argument('--pred_edges', type=int, default=1, help='!=0: use edges in dataset, 0: predict edges \
                                                                using L_0')
parser.add_argument('--random_seed', type=int, default=2024, help='size of common item be counted')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
parser.add_argument('--use_early_stop', type=bool, default=True, help='whether to use early stop')
parser.add_argument('--not_use_G', type=bool, default=False, help='whether to use graph-levl prediction layer')
parser.add_argument('--not_use_F', type=bool, default=False, help='whether to use field edges')
parser.add_argument('--not_use_E', type=bool, default=False, help='whether to learn two adjacency matrix')
args = parser.parse_args()

seed = args.random_seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset = Dataset2('/home/wy/gnn4rec_remote/data/', args.dataset, pred_edges=args.pred_edges)

num_feature = dataset.node_N()  #所有样本的特征总数
data_num = dataset.data_N()  #样本数量
field_num = dataset.field_N() #场数量

train_index = int(len(dataset) * 0.7)
test_index = int(len(dataset) * 0.85)
train_dataset = dataset[:train_index]
test_dataset = dataset[train_index:test_index]
val_dataset = dataset[test_index:]

del dataset

num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

show_loss = False
print(f"""
datast: {args.dataset}
vector embedding_dim: {args.embedding_dim}
batch_size: {args.batch_size}
lr: {args.lr}
l2: {args.l2_weight}
au: {args.au_weight}
l0: {args.l0_weight}
alpha: {args.alpha}
not_use_G: {args.not_use_G}
not_use_F: {args.not_use_F}
not_use_E: {args.not_use_E}
""")

datainfo = [train_loader, val_loader, test_loader, num_feature, field_num]
train(args, datainfo, show_loss, [len(train_dataset), len(val_dataset), len(test_dataset)])

