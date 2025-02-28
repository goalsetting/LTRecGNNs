import random

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.utils import InputType
from recbole.model.loss import BPRLoss, EmbLoss
from scipy.sparse import coo_matrix, diags
import numpy as np
import scipy.sparse as sp

from recbole.trainer import Trainer
from sklearn.cluster import KMeans

# from DeepCluster import DeepClusteringNetwork, clustering_loss

def delete_edges(interaction_matrix, tail_items_idx, head_users_idx, tail_users_idx, delete_percentage=0.5,
                 target_group='tail_users'):
    """
    删除长尾用户或头部用户与长尾商品之间的边

    :param interaction_matrix: 稀疏矩阵
    :param tail_items_idx: 长尾商品的索引列表
    :param head_users_idx: 头部用户的索引列表
    :param tail_users_idx: 长尾用户的索引列表
    :param delete_percentage: 删除边的百分比
    :param target_group: 删除的目标群体，可以是 'tail_users' 或 'head_users'
    :return: 删除边后的稀疏矩阵
    """

    # 获取稀疏矩阵的行（用户）和列（商品）索引
    row, col = interaction_matrix.row, interaction_matrix.col

    # 过滤出与长尾商品相关的边
    if target_group == 'tail_users':
        # 长尾用户与长尾商品的交互
        mask = np.isin(row, tail_users_idx) & np.isin(col, tail_items_idx)
    elif target_group == 'head_users':
        # 头部用户与长尾商品的交互
        mask = np.isin(row, head_users_idx) & np.isin(col, tail_items_idx)
    else:
        raise ValueError("target_group must be either 'tail_users' or 'head_users'")

    # 获取这些边的索引
    target_edges = np.where(mask)[0]

    # 随机选择要删除的边 int(len(target_edges) * delete_percentage)
    num_edges_to_delete = 1000
    edges_to_delete = random.sample(list(target_edges), num_edges_to_delete)

    # 删除选中的边
    new_row = np.delete(row, edges_to_delete)
    new_col = np.delete(col, edges_to_delete)

    # 构造删除后的稀疏矩阵
    new_interaction_matrix = sp.coo_matrix((np.ones(len(new_row)), (new_row, new_col)),
                                           shape=interaction_matrix.shape)

    return new_interaction_matrix

class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)
        self.config = config
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        self.interaction_matrix = delete_edges(self.interaction_matrix,config['tail_items'],config['head_users'],config['tail_users'])

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        ).to(self.device)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        ).to(self.device)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def calculate_uninteracted_scores(self, dataset, idx_headusers, idx_tailusers,head_item,tail_item):
        """
        计算头部用户与原始数据集中未交互的商品的预测分数总和
        :param model: RecBole模型（如 LightGCN）
        :param dataset: RecBole 数据集对象
        :param idx_headusers: 头部用户索引（一个 Tensor 或 NumPy 数组）
        :return: 头部用户未交互商品的分数总和
        """

        # 获取原始数据集中的交互矩阵（稀疏 COO 格式）
        interaction_matrix = torch.tensor(dataset.inter_matrix(form="coo").todense())


        # user_all_embeddings, item_all_embeddings = self.forward()
        user_all_embeddings = self.user_embedding.weight
        item_all_embeddings = self.item_embedding.weight
        u_embeddings = user_all_embeddings
        i_embeddings = item_all_embeddings
        scores = (F.normalize(u_embeddings) @ F.normalize(i_embeddings.t()))
        scores[interaction_matrix == 0] = 0
        scores_h = scores[idx_headusers]

        scores_t=scores[idx_tailusers]
        # scores_temp = scores[scores==0]

        scores_h[scores_h < 0] = 0
        scores_t[scores_t < 0] = 0

        # 获取前 K 个最大值及其对应的索引
        K = 20
        top_k_values, top_k_indices = torch.topk(scores_h, K)
        top_k_values2, top_k_indices2 = torch.topk(scores_t, K)
        # scores_h[scores_h < scores_temp.min()] = 0
        # scores_t[scores_t < scores_temp.min()] = 0
        # 计算预测分数总和
        total_score_hh = scores_h[:,head_item].sum() / (scores_h[:,head_item] > 0).sum()
        total_score_ht = scores_h[:, tail_item].sum() / (scores_h[:, tail_item] > 0).sum()
        total_score_th = scores_t[:,head_item].sum() / (scores_t[:,head_item] > 0).sum()
        total_score_tt = scores_t[:, tail_item].sum() / (scores_t[:, tail_item] > 0).sum()

        return total_score_hh, total_score_ht,total_score_th,total_score_tt

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)


class NewTrainer(Trainer):

    def __init__(self, config, model):
        super(NewTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx,show_progress):
        self.model.train()
        total_loss = 0.
        kmeans = KMeans(n_clusters=50, random_state=42)

        cluster_labels = kmeans.fit_predict(self.model.user_embedding.weight.detach().cpu().numpy())
        # cluster_labels = kmeans.fit_predict(self.user_embedding.weight.detach().cpu().numpy())
        # 更新聚类中心 (将结果重新移回 GPU)
        user_emb_add = torch.tensor(kmeans.cluster_centers_, device=self.device)
        for batch_idx, interaction in enumerate(train_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(interaction,user_emb_add,epoch_idx)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

class AlphaAdapter(nn.Module):
    def __init__(self, initial_alpha=500):
        super(AlphaAdapter, self).__init__()
        self.alpha = initial_alpha
        self.lastalpha = 0

    def forward(self, t):
        """
        计算时间 t 对应的 α 值
        α 随时间 t 的平方成反比递减：α(t) = initial_alpha / (t^2 + 1)
        增加 1 是为了避免除以 0 的问题
        """
        alpha_t = ((t + 1) / (2*self.alpha))**2
        return alpha_t

class CustomLightGCN2(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset, head, tail,a):
        super(CustomLightGCN2, self).__init__(config, dataset)
        self.a = a
        self.epochs = config['epochs']
        self.alpha = AlphaAdapter(initial_alpha=self.epochs)
        # 模型参数初始化
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        # 嵌入层定义
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Graph Transformer
        self.graph_transformer = GraphTransformer(self.embedding_size)

        # 初始化权重
        self.apply(xavier_uniform_initialization)
        # 在这里初始化BPRLoss
        self.loss = BPRLoss()
        self.inter_matrix = dataset.inter_matrix(form='coo')
        self.norm_adj = self.get_norm_adj_mat(self.inter_matrix)
        # self.deepcluster = DeepClusteringNetwork(self.embedding_size,self.embedding_size,100)
        # 指定输入类型
        self.device = config['device']
        self.head = torch.tensor(head).to(self.device)
        self.tail = torch.tensor(tail).to(self.device)
        self.to(self.device)

    def calculate_uninteracted_scores(self, dataset, idx_headusers, idx_tailusers,head_item,tail_item):
        """
        计算头部用户与原始数据集中未交互的商品的预测分数总和
        :param model: RecBole模型（如 LightGCN）
        :param dataset: RecBole 数据集对象
        :param idx_headusers: 头部用户索引（一个 Tensor 或 NumPy 数组）
        :return: 头部用户未交互商品的分数总和
        """

        # 获取原始数据集中的交互矩阵（稀疏 COO 格式）
        interaction_matrix = torch.tensor(dataset.inter_matrix(form="coo").todense())


        # user_all_embeddings, item_all_embeddings = self.forward()
        user_all_embeddings = self.user_embedding.weight
        item_all_embeddings = self.item_embedding.weight
        u_embeddings = user_all_embeddings
        i_embeddings = item_all_embeddings
        scores = (F.normalize(u_embeddings) @ F.normalize(i_embeddings.t()))
        scores[interaction_matrix == 0] = 0
        scores_h = scores[idx_headusers]

        scores_t=scores[idx_tailusers]
        # scores_temp = scores[scores==0]

        scores_h[scores_h < 0] = 0
        scores_t[scores_t < 0] = 0
        # 计算预测分数总和
        total_score_hh = scores_h[:,head_item].sum() / (scores_h[:,head_item] > 0).sum()
        total_score_ht = scores_h[:, tail_item].sum() / (scores_h[:, tail_item] > 0).sum()
        total_score_th = scores_t[:,head_item].sum() / (scores_t[:,head_item] > 0).sum()
        total_score_tt = scores_t[:, tail_item].sum() / (scores_t[:, tail_item] > 0).sum()

        return total_score_hh, total_score_ht,total_score_th,total_score_tt

    def contrast_loss(self,head_emb,tail_emb):
        head_emb = F.normalize(head_emb[self.head])
        tail_emb = F.normalize(tail_emb[self.head])
        return -(head_emb@tail_emb.t()).diag().mean()

    def get_sampled_norm_adj_mat(self, inter_matrix, idx_data, device='cuda'):
        n_users = idx_data.shape[0]
        n_items = self.n_items
        n_nodes = n_users + n_items

        # Convert dataset interaction matrix to COO format

        # Extract rows and columns of the user-item interaction matrix
        user_indices = torch.tensor(inter_matrix.row, dtype=torch.long, device=device)
        item_indices = torch.tensor(inter_matrix.col, dtype=torch.long, device=device) + n_users

        # 过滤出采样的用户索引的交互
        sampled_mask = torch.isin(user_indices, torch.tensor(idx_data, device=device))
        sampled_user_indices = user_indices[sampled_mask]
        sampled_item_indices = item_indices[sampled_mask]
        # Concatenate user-item and item-user interactions
        row = torch.cat([sampled_user_indices, sampled_item_indices])
        col = torch.cat([sampled_item_indices, sampled_user_indices])

        # Create data tensor (all ones)
        data = torch.ones(len(row), dtype=torch.float32, device=device)

        # # Create sampled adjacency matrix in COO format using PyTorch
        sampled_adj_matrix = torch.sparse_coo_tensor(
            torch.stack([row, col]),
            data,
            (n_nodes, n_nodes),
            device=device
        )

        # Sum rows (degree of nodes)
        row_sum = torch.sparse.sum(sampled_adj_matrix, dim=1).to_dense()

        # Inverse square root of the row sum
        d_inv = torch.pow(row_sum, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.0

        # Create sparse diagonal matrix for D^(-1/2)
        d_inv_indices = torch.arange(n_nodes, device=device)
        d_inv_diag = torch.sparse_coo_tensor(
            torch.stack([d_inv_indices, d_inv_indices]),
            d_inv,
            (n_nodes, n_nodes),
            device=device
        )

        # Normalize the adjacency matrix using sparse matrix multiplication
        norm_adj_tmp = torch.sparse.mm(d_inv_diag, sampled_adj_matrix)  # D^(-1/2) * A
        norm_adj_matrix = torch.sparse.mm(norm_adj_tmp, d_inv_diag)  # (D^(-1/2) * A) * D^(-1/2)

        return norm_adj_matrix

    def get_norm_adj_mat(self, inter_matrix, device='cuda'):
        n_users = self.n_users
        n_items = self.n_items
        n_nodes = n_users + n_items
        # Convert dataset interaction matrix to COO format

        # Extract rows and columns of the user-item interaction matrix
        user_indices = torch.tensor(inter_matrix.row, dtype=torch.long, device=device)
        item_indices = torch.tensor(inter_matrix.col, dtype=torch.long, device=device) + n_users

        # 过滤出采样的用户索引的交互
        sampled_user_indices = user_indices
        sampled_item_indices = item_indices
        # Concatenate user-item and item-user interactions
        row = torch.cat([sampled_user_indices, sampled_item_indices])
        col = torch.cat([sampled_item_indices, sampled_user_indices])

        # Create data tensor (all ones)
        data = torch.ones(len(row), dtype=torch.float32, device=device)

        # # Create sampled adjacency matrix in COO format using PyTorch
        sampled_adj_matrix = torch.sparse_coo_tensor(
            torch.stack([row, col]),
            data,
            (n_nodes, n_nodes),
            device=device
        )

        # Sum rows (degree of nodes)
        row_sum = torch.sparse.sum(sampled_adj_matrix, dim=1).to_dense()

        # Inverse square root of the row sum
        d_inv = torch.pow(row_sum, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.0

        # Create sparse diagonal matrix for D^(-1/2)
        d_inv_indices = torch.arange(n_nodes, device=device)
        d_inv_diag = torch.sparse_coo_tensor(
            torch.stack([d_inv_indices, d_inv_indices]),
            d_inv,
            (n_nodes, n_nodes),
            device=device
        )

        # Normalize the adjacency matrix using sparse matrix multiplication
        norm_adj_tmp = torch.sparse.mm(d_inv_diag, sampled_adj_matrix)  # D^(-1/2) * A
        norm_adj_matrix = torch.sparse.mm(norm_adj_tmp, d_inv_diag)  # (D^(-1/2) * A) * D^(-1/2)

        return norm_adj_matrix

    import torch

    def update_and_normalize_adj_matrix(self, sparse_tensor, item_adj_matrix):
        n_users = self.n_users

        # 获取 sparse_tensor 和 item_adj_matrix 的行、列和数据
        row_col, data = sparse_tensor._indices(), sparse_tensor._values()
        data = torch.ones_like(data).to(self.device)

        row_col_item, item_data = item_adj_matrix._indices(), item_adj_matrix._values()
        # 调整 item_adj_matrix 的索引位置，并获取其行、列和数据
        item_row = row_col_item[0] + n_users
        item_col = row_col_item[1] + n_users

        # 创建新的 indices 和 values 列表，用于构建更新后的稀疏张量
        new_row = torch.cat([row_col[0], item_row])
        new_col = torch.cat([row_col[1], item_col])
        new_data = torch.cat([data, item_data])

        # 构建新的稀疏邻接矩阵
        new_sparse_tensor = torch.sparse_coo_tensor(torch.stack([new_row, new_col]), new_data, sparse_tensor.shape).to(
            sparse_tensor.device)

        # 对新的邻接矩阵进行归一化
        # 计算度矩阵的逆平方根
        degree = torch.sparse.sum(new_sparse_tensor, dim=1).to_dense()
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

        # 构建稀疏对角矩阵
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to(sparse_tensor.device)

        # 将对角矩阵转换为稀疏格式
        d_mat_inv_sqrt_sparse = d_mat_inv_sqrt.to_sparse()

        # D^(-1/2) * A * D^(-1/2)
        normalized_sparse_tensor = torch.sparse.mm(d_mat_inv_sqrt_sparse, new_sparse_tensor)
        normalized_sparse_tensor = torch.sparse.mm(normalized_sparse_tensor, d_mat_inv_sqrt_sparse)

        return normalized_sparse_tensor

    def headView(self,t):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        if t/self.epochs > 0:
            # 将 list 转换为 PyTorch 张量
            head_indices = self.head
            # tail_indices = torch.tensor(self.tail)

            norm_item = F.normalize(item_emb)
            cosine_similarity = norm_item[head_indices] @ norm_item[head_indices].t()

            # 找出余弦相似度大于0.8的索引
            # adj_matrix = torch.zeros((self.n_items, self.n_items)).to(self.device)
            mask = cosine_similarity > self.a
            cosine_similarity[cosine_similarity>0]=1
            # 只保留符合条件的位置
            masked_similarity = cosine_similarity * mask

            # 获取非零元素的行和列索引
            head_nonzero_idx, tail_nonzero_idx = torch.nonzero(masked_similarity, as_tuple=True)

            # 获取这些非零元素的值
            values = masked_similarity[head_nonzero_idx, tail_nonzero_idx]

            # 调整这些索引的维度以对称化邻接矩阵
            indices = torch.cat([
                torch.stack([head_indices[head_nonzero_idx], head_indices[tail_nonzero_idx]]),  # 原始的非零项
                torch.stack([head_indices[tail_nonzero_idx], head_indices[head_nonzero_idx]])  # 对称项
            ], dim=1)

            # 拼接对应的值来形成完整的邻接矩阵
            values = torch.cat([values, values])

            # 构建稀疏邻接矩阵
            adj_matrix = torch.sparse_coo_tensor(indices, values, (self.n_items, self.n_items)).to(self.device)

            norm_adj = self.update_and_normalize_adj_matrix(self.norm_adj,adj_matrix)
        else:
            norm_adj = self.norm_adj
        all_embeddings = torch.cat([user_emb, item_emb], dim=0)
        embeddings_list = [all_embeddings]

        # 消息传递与聚合过程
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)

        return final_embeddings

    def tailView(self,batch_idx,user_emb_add,t):
        user_emb = self.user_embedding.weight[batch_idx]
        if user_emb_add!=[]:
            user_emb_extend = torch.cat((user_emb, user_emb_add), dim=0)
        else:
            user_emb_extend = user_emb
        item_emb = self.item_embedding.weight

        item_emb = self.graph_transformer(user_emb_extend, item_emb)

        # norm_adj = self.get_sampled_norm_adj_mat(self.inter_matrix,batch_idx)
        user_emb = self.user_embedding.weight

        if t/self.epochs > 0:
            # 将 list 转换为 PyTorch 张量
            head_indices = self.head
            tail_indices = self.tail

            norm_item = F.normalize(item_emb)
            cosine_similarity = norm_item[head_indices] @ norm_item[tail_indices].t()

            # 找出余弦相似度大于0.8的索引
            # adj_matrix = torch.zeros((self.n_items, self.n_items)).to(self.device)
            mask = cosine_similarity > self.a
            cosine_similarity[cosine_similarity > 0] = 1
            # 只保留符合条件的位置
            masked_similarity = cosine_similarity * mask

            # 获取非零元素的行和列索引
            head_nonzero_idx, tail_nonzero_idx = torch.nonzero(masked_similarity, as_tuple=True)

            # 获取这些非零元素的值
            values = masked_similarity[head_nonzero_idx, tail_nonzero_idx]

            # 调整这些索引的维度以对称化邻接矩阵
            indices = torch.cat([
                torch.stack([head_indices[head_nonzero_idx], tail_indices[tail_nonzero_idx]]),  # 原始的非零项
                torch.stack([tail_indices[tail_nonzero_idx], head_indices[head_nonzero_idx]])  # 对称项
            ], dim=1)

            # 拼接对应的值来形成完整的邻接矩阵
            values = torch.cat([values, values])

            # 构建稀疏邻接矩阵
            adj_matrix = torch.sparse_coo_tensor(indices, values, (self.n_items, self.n_items)).to(self.device)

            norm_adj = self.update_and_normalize_adj_matrix(self.norm_adj,adj_matrix)
        else:
            norm_adj = self.norm_adj
        all_embeddings0 = torch.cat([user_emb, item_emb], dim=0)
        embeddings_list = [all_embeddings0]
        all_embeddings = all_embeddings0
        # 消息传递与聚合过程
        for layer in range(self.n_layers-1):
            all_embeddings = torch.sparse.mm(norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        # 最终嵌入
            # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)
        return final_embeddings

    def forward(self, batch_idx, user_emb_add, t=1000):
        head_emb = self.headView(t)
        tail_emb = self.tailView(batch_idx,user_emb_add,t)
        if t == 1000:
            t = self.epochs
        final_embeddings = (1-self.alpha(t))*head_emb + self.alpha(t)*tail_emb
        self.alpha.lastalpha = self.alpha(t)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items]) #batch_idx.shape[0]
        _, item_head_embeddings = torch.split(head_emb, [self.n_users, self.n_items])
        _, item_tail_embeddings = torch.split(tail_emb, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, item_head_embeddings, item_tail_embeddings

    def eval_forward(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        item_emb = self.graph_transformer(user_emb, item_emb)
        # norm_adj = self.get_sampled_norm_adj_mat(self.inter_matrix,batch_idx)
        # 将 list 转换为 PyTorch 张量
        head_indices = self.head
        tail_indices = self.tail

        norm_item = F.normalize(item_emb)
        cosine_similarity = norm_item[head_indices] @ norm_item[tail_indices].t()

        # 找出余弦相似度大于0.8的索引
        mask = cosine_similarity > self.a
        cosine_similarity[cosine_similarity > 0] = 1
        # 只保留符合条件的位置
        masked_similarity = cosine_similarity * mask

        # 获取非零元素的行和列索引
        head_nonzero_idx, tail_nonzero_idx = torch.nonzero(masked_similarity, as_tuple=True)

        # 获取这些非零元素的值
        values = masked_similarity[head_nonzero_idx, tail_nonzero_idx]

        # 调整这些索引的维度以对称化邻接矩阵
        indices = torch.cat([
            torch.stack([head_indices[head_nonzero_idx], tail_indices[tail_nonzero_idx]]),  # 原始的非零项
            torch.stack([tail_indices[tail_nonzero_idx], head_indices[head_nonzero_idx]])  # 对称项
        ], dim=1)

        # 拼接对应的值来形成完整的邻接矩阵
        values = torch.cat([values, values])

        # 构建稀疏邻接矩阵
        adj_matrix = torch.sparse_coo_tensor(indices, values, (self.n_items, self.n_items)).to(self.device)


        norm_adj = self.update_and_normalize_adj_matrix(self.norm_adj, adj_matrix)

        all_embeddings0 = torch.cat([user_emb, item_emb], dim=0)


        embeddings_list = [all_embeddings0]
        all_embeddings = all_embeddings0
        # 消息传递与聚合过程
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)

        # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)

        # head_emb = self.headView(10000)
        #
        # final_embeddings = (1 - self.alpha.lastalpha) * head_emb + self.alpha.lastalpha * final_embeddings

        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction,user_emb_add,t):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]



        user_emb, item_emb, item_head_emb, item_tail_emb= self.forward(user,user_emb_add,t)

        user_emb = user_emb[user]
        pos_item_emb = item_emb[pos_item]
        neg_item_emb = item_emb[neg_item]

        reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                              pos_item_emb.norm(2).pow(2) +
                              neg_item_emb.norm(2).pow(2)) / float(len(user))

        # # 计算正负样本得分
        # pos_score = (user_emb @ pos_item_emb.t()).sum(dim=1)
        # neg_score = (user_emb @ neg_item_emb.t()).sum(dim=1)
        # 计算正负样本得分
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)

        if t / self.epochs > 0.2:
            # 计算BPR损失
            loss = self.loss(pos_score, neg_score) + 0 * self.contrast_loss(item_head_emb, item_tail_emb) + 0*reg_loss
        else:
            loss = self.loss(pos_score, neg_score) + 0*reg_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]

        user_emb, item_emb,_,_ = self.forward()

        user_emb = user_emb[user]
        item_emb = item_emb[self.head]

        # 计算预测评分
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

    def recall_at_k(self,pred_scores, ground_truth, k):
        """
        Calculate recall@k given predicted scores and ground truth.

        Parameters:
        - pred_scores: List or numpy array of predicted scores for items.
        - ground_truth: List or numpy array indicating ground truth labels (1 for relevant, 0 for non-relevant).
        - k: Top-k value.

        Returns:
        - recall: Recall@k value.
        """
        # 对预测分数进行排序，并获取top-k的索引
        top_k_indices = np.argsort(pred_scores)[::-1][:k]

        # 计算在top-k中实际为1（即真实为relevant的item）的数量
        num_relevant_in_top_k = np.sum(ground_truth[top_k_indices] == 1)

        # 计算总的relevant item数量
        total_relevant = np.sum(ground_truth == 1)

        # 计算recall@k
        recall = num_relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

        return recall

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_emb, item_emb = self.eval_forward()

        user_emb = user_emb[user]

        # 计算全排序预测评分
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

class CustomLightGCN(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset,head):
        super(CustomLightGCN, self).__init__(config, dataset)
        self.head = head
        # 模型参数初始化
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        # 嵌入层定义
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Graph Transformer
        self.graph_transformer = GraphTransformer(self.embedding_size)

        self.graph_transformer2 = GraphTransformer(self.embedding_size)

        self.graph_transformer3 = GraphTransformer(self.embedding_size)

        # 初始化权重
        self.apply(xavier_uniform_initialization)
        # 在这里初始化BPRLoss
        self.loss = BPRLoss()

        # 指定输入类型
        self.device = config['device']
        self.to(self.device)

        # 构建归一化的邻接矩阵
        self.norm_adj_matrix = self.get_norm_adj_mat(dataset)
        self.norm_adj_matrix = self.norm_adj_matrix.to(self.device)

    def symmetric_normalize_adjacency(self,adj):
        # Calculate degree matrix
        degrees = torch.sum(adj, dim=1)  # Sum along rows to get degrees
        degree_sqrt_inv = torch.pow(degrees, -0.5)
        degree_sqrt_inv[torch.isinf(degree_sqrt_inv)] = 0.  # Handle division by zero (if any)

        # Create diagonal degree matrix
        D_sqrt_inv = torch.diag(degree_sqrt_inv)

        # Symmetrically normalize adjacency matrix: A' = D^-0.5 * A * D^-0.5
        adj = adj  # Convert to dense matrix for computation
        adj = D_sqrt_inv @ adj @ D_sqrt_inv

        return adj

    def get_ego_embeddings(self,user_emb,item_emb):
        item_emb = self.graph_transformer(user_emb, item_emb)
        temp_item = F.normalize(item_emb)
        cos_item = temp_item @ temp_item.t()
        tha = 0.8
        cos_item[cos_item < tha] = 0
        cos_item[cos_item >= tha] = 1
        cos_item.fill_diagonal_(0)
        adj_adjust = self.symmetric_normalize_adjacency(cos_item)
        # print(adj_adjust.max())
        if item_emb.sum()!=0:
            item_emb2 = item_emb + torch.spmm(adj_adjust, item_emb)
        else:
            item_emb2 = item_emb
        return torch.cat([user_emb, item_emb], dim=0), torch.cat([user_emb, item_emb2], dim=0),user_emb,item_emb,item_emb2

    def get_norm_adj_mat(self, dataset):

        n_nodes = self.n_users + self.n_items
        row = np.concatenate([dataset.inter_matrix(form='coo').row, dataset.inter_matrix(form='coo').col + self.n_users])
        col = np.concatenate([dataset.inter_matrix(form='coo').col + self.n_users, dataset.inter_matrix(form='coo').row])
        data = np.ones(len(row))

        adj_matrix = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        row_sum = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv = np.power(row_sum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_matrix)
        # Assuming norm_adj_tmp and d_mat_inv are defined earlier
        norm_adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # Convert to COO format if not already in COO
        if not scipy.sparse.isspmatrix_coo(norm_adj_matrix):
            norm_adj_matrix = norm_adj_matrix.tocoo()

        # Construct sparse tensor in PyTorch
        row = torch.LongTensor(norm_adj_matrix.row)
        col = torch.LongTensor(norm_adj_matrix.col)
        data = torch.FloatTensor(norm_adj_matrix.data)
        shape = torch.Size(norm_adj_matrix.shape)

        sparse_tensor = torch.sparse_coo_tensor(torch.stack([row, col]), data, shape)
        return sparse_tensor

    def forward(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_embeddings0 = torch.cat([user_emb, item_emb], dim=0)


        embeddings_list = [all_embeddings0]
        all_embeddings, all_embeddings2, _, _, _ = self.get_ego_embeddings(user_emb, item_emb)
        # 消息传递与聚合过程
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            all_embeddings2 = torch.sparse.mm(self.norm_adj_matrix, all_embeddings2)
            embeddings_list.append(all_embeddings+all_embeddings2)

        # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)


        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
        # user_embeddings = self.user_embedding(user)
        # item_embeddings = self.item_embedding(item)
        #
        # # Graph Transformer全局学习
        # global_user_embeddings, global_item_embeddings = self.graph_transformer(user_embeddings, item_embeddings)
        #
        # # LightGCN局部约束学习
        # for conv in self.convs:
        #     local_user_embeddings, local_item_embeddings = conv(global_user_embeddings, global_item_embeddings)
        #
        # # 对比学习（头部商品一致性约束）
        # head_user_embeddings = local_user_embeddings[:len(head_users)]
        # head_item_embeddings = local_item_embeddings[:len(head_items)]
        # contrastive_loss = self.contrastive_loss(head_user_embeddings, head_item_embeddings)
        #
        # # 融合输出
        # final_user_embeddings = global_user_embeddings + local_user_embeddings
        # final_item_embeddings = global_item_embeddings + local_item_embeddings
        #
        # # 计算预测得分
        # scores = torch.mul(final_user_embeddings, final_item_embeddings).sum(dim=1)

        # return scores, contrastive_loss

    def contrastive_loss(self, user_embeddings, item_embeddings):
        # 计算对比损失
        pass  # 实现对比损失的计算方法

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_emb, item_emb = self.forward()

        user_emb = user_emb[user]
        pos_item_emb = item_emb[pos_item]
        neg_item_emb = item_emb[neg_item]

        # 计算正负样本得分
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)

        # 计算BPR损失
        loss = self.loss(pos_score, neg_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]

        user_emb, item_emb = self.forward()

        user_emb = user_emb[user]
        item_emb = item_emb[self.head]

        # 计算预测评分
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

    def recall_at_k(self,pred_scores, ground_truth, k):
        """
        Calculate recall@k given predicted scores and ground truth.

        Parameters:
        - pred_scores: List or numpy array of predicted scores for items.
        - ground_truth: List or numpy array indicating ground truth labels (1 for relevant, 0 for non-relevant).
        - k: Top-k value.

        Returns:
        - recall: Recall@k value.
        """
        # 对预测分数进行排序，并获取top-k的索引
        top_k_indices = np.argsort(pred_scores)[::-1][:k]

        # 计算在top-k中实际为1（即真实为relevant的item）的数量
        num_relevant_in_top_k = np.sum(ground_truth[top_k_indices] == 1)

        # 计算总的relevant item数量
        total_relevant = np.sum(ground_truth == 1)

        # 计算recall@k
        recall = num_relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

        return recall

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_emb, item_emb = self.forward()

        user_emb = user_emb[user]

        # 计算全排序预测评分
        scores = torch.matmul(user_emb, item_emb.t())
        return scores


class GraphTransformer(nn.Module):
    def __init__(self, embedding_size):
        super(GraphTransformer, self).__init__()
        self.embedding_size = embedding_size
        # 这里可以定义多层Graph Transformer的层
        self.gloabal_layers = EncoderLayer(embedding_size, 4, 0.5)

    def forward(self, user_embeddings, item_embeddings):
        # 实现Graph Transformer的前向传播
        global_item_embeddings = self.gloabal_layers(user_embeddings, item_embeddings)

        return global_item_embeddings

class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count

        self.keys = nn.Linear(in_channels, key_channels)
        self.queries = nn.Linear(in_channels, key_channels)
        self.reprojection = nn.Linear(key_channels*head_count, key_channels)

    def forward(self, user_embeddings, item_embeddings):
        keys = self.keys(user_embeddings)
        queries = self.queries(item_embeddings)
        head_key_channels = self.key_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:,i * head_key_channels: (i + 1) * head_key_channels], dim=0)
            query = F.softmax(queries[:,i * head_key_channels: (i + 1) * head_key_channels], dim=1)
            context = F.relu((key @ query.transpose(0, 1))-0.2)
            attended_value = context.t() @ user_embeddings
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        return attention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.effectattn = EfficientAttention(in_channels = d_model, key_channels = 32, head_count =heads, value_channels = 32)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, y):
        x_pre = self.effectattn(x, y)
        y = y + self.dropout_1(x_pre)
        return y