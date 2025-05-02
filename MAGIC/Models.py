import os
import numpy as np
from time import time
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from NoiseAttnHGCN import NoiseAttnHGCN
from contrast import Contrast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_scatter import scatter_sum, scatter_mean

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph

args = parse_args()


def _adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score, keep_rate):
    _, least_attn_edge_id = torch.topk(
        -edge_attn_score, int((1 - keep_rate) * edge_attn_score.shape[0]), sorted=False
    )
    cl_kg_mask = torch.ones_like(edge_attn_score).bool()
    cl_kg_mask[least_attn_edge_id] = False
    cl_kg_edge = edge_index[:, cl_kg_mask]
    cl_kg_type = edge_type[cl_kg_mask]
    return cl_kg_edge, cl_kg_type


def _adaptive_ui_drop_cl(
    item_attn_mean, inter_edge, inter_edge_w, keep_rate=0.7, samp_func="torch"
):
    inter_attn_prob = item_attn_mean[inter_edge[1]]
    # add gumbel noise
    noise = -torch.log(-torch.log(torch.rand_like(inter_attn_prob)))
    """ prob based drop """
    inter_attn_prob = inter_attn_prob + noise
    inter_attn_prob = F.softmax(inter_attn_prob, dim=0)

    if samp_func == "np":
        # we observed abnormal behavior of torch.multinomial on mind
        sampled_edge_idx = np.random.choice(
            np.arange(inter_edge_w.shape[0]),
            size=int(keep_rate * inter_edge_w.shape[0]),
            replace=False,
            p=inter_attn_prob.cpu().numpy(),
        )
    else:
        sampled_edge_idx = torch.multinomial(
            inter_attn_prob, int(keep_rate * inter_edge_w.shape[0]), replacement=False
        )

    return inter_edge[:, sampled_edge_idx], inter_edge_w[sampled_edge_idx] / keep_rate


def _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    topk_egde_id = topk_egde_id.cpu().numpy()
    topk_mask = np.zeros(n_edges, dtype=bool)
    topk_mask[topk_egde_id] = True
    # add another group of random mask
    random_indices = np.random.choice(
        n_edges, size=topk_egde_id.shape[0], replace=False
    )
    random_mask = np.zeros(n_edges, dtype=bool)
    random_mask[random_indices] = True
    # combine two masks
    mask = topk_mask | random_mask

    remain_edge_index = edge_index[:, ~mask]
    remain_edge_type = edge_type[~mask]
    masked_edge_index = edge_index[:, mask]
    masked_edge_type = edge_type[mask]

    return (
        remain_edge_index,
        remain_edge_type,
        masked_edge_index,
        masked_edge_type,
        mask,
    )


def _edge_sampling(edge_index, edge_type, samp_rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * samp_rate), replace=False
    )
    return edge_index[:, random_indices], edge_type[random_indices]


def _sparse_dropout(i, v, keep_rate=0.5):
    noise_shape = i.shape[1]

    random_tensor = keep_rate
    # the drop rate is 1 - keep_rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    i = i[:, dropout_mask]
    v = v[dropout_mask] / keep_rate

    return i, v


def _relation_aware_edge_sampling(edge_index, edge_type, n_relations, samp_rate=0.5):
    # exclude interaction
    for i in range(n_relations - 1):
        edge_index_i, edge_type_i = _edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate
        )
        if i == 0:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
        else:
            edge_index_sampled = torch.cat([edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat([edge_type_sampled, edge_type_i], dim=0)
    return edge_index_sampled, edge_type_sampled


class MMSSL(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        n_relations,
        n_entities,
        embedding_dim,
        weight_size,
        dropout_list,
        kg_cf_graph,
        image_feats,
        text_feats,
        noise=1.5,
        L=-1,
        L1=-1,
        use_CL=True,
        use_AL=True,
    ):

        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_relations = n_relations
        self.n_entities = n_entities
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.node_dropout = 1
        self.node_dropout_rate = 0.5
        self.mess_dropout = 1
        self.mess_dropout_rate = 0.1
        self.mae_msize = 256
        self.mae_coef = 0.1
        self.cl_coef = 0.01
        self.use_CL=True
        self.use_AL=True

        self.L=L if L>=0 else 2
        self.layers=L1 if L1>=0 else 1

        self.graph = kg_cf_graph
        self.edge_index, self.edge_type = self._get_edges(self.graph)

        self.image_trans = nn.Linear(image_feats.shape[1], args.embed_size)
        self.text_trans = nn.Linear(text_feats.shape[1], args.embed_size)
        self.fuse = CrossAttention(self.embedding_dim)
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)
        self.encoder = nn.ModuleDict()
        self.encoder["image_encoder"] = self.image_trans
        self.encoder["text_encoder"] = self.text_trans
        self.kg_gcn = NoiseAttnHGCN(
            channel=self.embedding_dim,
            n_hops=self.L,
            # n_users=self.n_users,
            n_relations=self.n_relations,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
            noise_scale=noise,
        )
        self.contrast_fn = Contrast(self.embedding_dim, tau=1.0)

        self.common_trans = nn.Linear(args.embed_size, args.embed_size)
        nn.init.xavier_uniform_(self.common_trans.weight)
        self.align = nn.ModuleDict()
        self.align["common_trans"] = self.common_trans

        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        self.entity_id_embedding = nn.Embedding(
            n_entities - n_items, self.embedding_dim
        )
        self.entity_mm_embedding = nn.Embedding(
            n_entities - n_items, self.embedding_dim
        )
        self.item_kg_emb_temp=nn.Parameter(torch.zeros((image_feats.shape[0],self.embedding_dim)),)
        nn.init.xavier_uniform_(self.item_kg_emb_temp)

        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        nn.init.xavier_uniform_(self.entity_id_embedding.weight)
        self.image_feats = torch.tensor(image_feats).float()
        self.text_feats = torch.tensor(text_feats).float()
        self.image_embedding = nn.Embedding.from_pretrained(
            torch.Tensor(image_feats), freeze=False
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            torch.Tensor(text_feats), freeze=False
        )

        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(args.embed_size)
        self.tau = 0.5

        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict(
            {
                "w_q": nn.Parameter(
                    initializer(torch.empty([args.embed_size, args.embed_size]))
                ),
                "w_k": nn.Parameter(
                    initializer(torch.empty([args.embed_size, args.embed_size]))
                ),
                "w_v": nn.Parameter(
                    initializer(torch.empty([args.embed_size, args.embed_size]))
                ),
                # "w_self_attention_item": nn.Parameter(
                #     initializer(torch.empty([args.embed_size, args.embed_size]))
                # ),
                # "w_self_attention_user": nn.Parameter(
                #     initializer(torch.empty([args.embed_size, args.embed_size]))
                # ),
                # "w_self_attention_cat": nn.Parameter(
                #     initializer(
                #         torch.empty([args.head_num * args.embed_size, args.embed_size])
                #     )
                # ),
            }
        )
        self.embedding_dict = {"user": {}, "item": {}}
        self.w_mm=nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.n_items, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.w_mm.data = F.softmax(self.w_mm.data, dim=1)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().cuda(), type.long().cuda()

    def mm(self, x, y):
        if args.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            losses.append(
                -torch.log(
                    between_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                    / (
                        refl_sim.sum(1)
                        + between_sim.sum(1)
                        - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                    )
                )
            )

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.0
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.0
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(
            np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64)
        )  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)

        return (
            torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()
        )  #

    def para_dict_to_tenser(self, para_dict):
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors

    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):

        q = self.para_dict_to_tenser(embedding_t)
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], args.embed_size / args.head_num

        Q = torch.matmul(q, trans_w["w_q"])
        K = torch.matmul(k, trans_w["w_k"])
        V = v

        Q = Q.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)
        K = Q.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)

        Q = torch.unsqueeze(Q, 2)
        K = torch.unsqueeze(K, 1)
        V = torch.unsqueeze(V, 1)

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))
        att = torch.sum(att, dim=-1)
        att = torch.unsqueeze(att, dim=-1)
        att = F.softmax(att, dim=2)

        Z = torch.mul(att, V)
        Z = torch.sum(Z, dim=2)

        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict["w_self_attention_cat"])

        # args.model_cat_rate * F.normalize(Z, p=2, dim=2)
        return Z, att.detach()

    def forward(
        self,
        ui_graph,
        iu_graph,
        mm_ui_graph,
        mm_iu_graph,
        predict_mode=False
    ):
        use_MM=False
        use_KG=True
        if use_MM:
            # MM
            image_feats = self.dropout(self.image_trans(self.image_feats.cuda()))
            text_feats = self.dropout(self.text_trans(self.text_feats.cuda()))
            # mm_feats = mm_item_feats = self.fuse(
            #     torch.stack((image_feats, text_feats), dim=0)
            # )
            mm_feats=mm_item_feats=image_feats+text_feats
            # mm_feats=mm_item_feats=torch.matmul(torch.stack((image_feats, text_feats), dim=2),self.w_mm).squeeze()
        else:
            mm_feats=self.item_kg_emb_temp

        entity_emb = torch.cat(
            (mm_feats, self.entity_mm_embedding.weight), dim=0
        )
        
        if use_KG:
            edge_index, edge_type = _relation_aware_edge_sampling(
                self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate
            )

            entity_mmkg_emb = self.kg_gcn(
                entity_emb,
                edge_index,
                edge_type,
                mess_dropout=self.mess_dropout,
                predict_mode=predict_mode
            )[:self.n_items]
        else:
            entity_mmkg_emb=entity_emb[:self.n_items]

        mm_user_feats_list = []
        mm_item_feats_list = []
        mm_user_id_list = []
        mm_item_id_list = []
        mm_user_feats_list.append(self.mm(ui_graph, entity_mmkg_emb))
        mm_item_feats_list.append(self.mm(iu_graph, mm_user_feats_list[-1]))
        if self.use_AL:
            mm_user_id_list.append(self.mm(mm_ui_graph, self.item_id_embedding.weight))
            mm_item_id_list.append(self.mm(mm_iu_graph, self.user_id_embedding.weight))
        else:
            mm_user_id_list.append(self.user_id_embedding.weight)
            mm_item_id_list.append(self.item_id_embedding.weight)

        for i in range(self.layers-1):
            mm_user_feats_list.append(self.mm(ui_graph, mm_item_feats_list[-1]))
            mm_item_feats_list.append(self.mm(iu_graph, mm_user_feats_list[-2]))
            mm_user_id_list.append(self.mm(mm_ui_graph, mm_item_id_list[-1]))
            mm_item_id_list.append(self.mm(mm_iu_graph, mm_user_id_list[-2]))

        mm_user_feats=torch.stack(mm_user_feats_list).mean(dim=0)
        mm_item_feats=torch.stack(mm_item_feats_list).mean(dim=0)
        mm_user_id=torch.stack(mm_user_id_list).mean(dim=0)
        mm_item_id=torch.stack(mm_item_id_list).mean(dim=0)

        # continue
        user_emb = self.embedding_dict["user"]["mm"] = mm_user_id
        item_emb = self.embedding_dict["item"]["mm"] = mm_item_id

        # user_z, _ = self.multi_head_self_attention(
        #     self.weight_dict, self.embedding_dict["user"], self.embedding_dict["user"]
        # )
        # item_z, _ = self.multi_head_self_attention(
        #     self.weight_dict, self.embedding_dict["item"], self.embedding_dict["item"]
        # )
        # user_emb = user_z.mean(0)
        # item_emb = item_z.mean(0)
        u_g_embeddings = self.user_id_embedding.weight + args.id_cat_rate * F.normalize(
            user_emb, p=2, dim=1
        )
        i_g_embeddings = self.item_id_embedding.weight + args.id_cat_rate * F.normalize(
            item_emb, p=2, dim=1
        )

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):
            if i == (self.n_ui_layers - 1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))

            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)

        u_g_embeddings = u_g_embeddings + args.model_cat_rate * F.normalize(
            mm_user_feats, p=2, dim=1
        )
        i_g_embeddings = i_g_embeddings + args.model_cat_rate * F.normalize(
            mm_item_feats, p=2, dim=1
        )

        return (
            u_g_embeddings,
            i_g_embeddings,
            mm_item_feats,
            mm_user_feats,
            u_g_embeddings,
            i_g_embeddings,
            mm_user_id,
            mm_item_id,
        )

    def _convert_sp_mat_to_tensor(self, X):
        # coo = X.tocoo()
        # 获取稀疏张量的索引和值
        indices = X._indices()
        values = X._values()

        # 提取行和列索引
        rows = indices[0].cpu().numpy()
        cols = indices[1].cpu().numpy()
        data = values.cpu().numpy()

        # 创建 SciPy 的 coo_matrix
        coo = sp.coo_matrix((data, (rows, cols)), shape=X.shape)
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return i.cuda(), v.cuda()

    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = -torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, int(dim / 8)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim / 8)),
            nn.Dropout(args.G_drop1),
            nn.Linear(int(dim / 8), int(dim / 16)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim / 16)),
            nn.Dropout(args.G_drop2),
            nn.Linear(int(dim / 16), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = 100 * self.net(x.float())
        return output.view(-1)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(CrossAttention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x_qkv):
        n_emb, b, d = x_qkv.shape
        h = self.heads

        # Compute queries, keys, values
        k = (
            self.to_k(x_qkv).view(n_emb, b, h, -1).permute(1, 2, 0, 3)
        )  # shape: (b, h, n_emb, d)
        v = (
            self.to_v(x_qkv).view(n_emb, b, h, -1).permute(1, 2, 0, 3)
        )  # shape: (b, h, n_emb, d)
        q = (
            self.to_q(x_qkv[0].unsqueeze(0)).view(1, b, h, -1).permute(1, 2, 0, 3)
        )  # shape: (b, h, 1, d)

        # Compute attention scores
        dots = (
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )  # shape: (b, h, 1, n_emb)
        attn = dots.softmax(dim=-1)  # shape: (b, h, 1, n_emb)

        # Compute attention output
        out = torch.matmul(attn, v)  # shape: (b, h, 1, d)
        out = (
            out.permute(0, 2, 1, 3).contiguous().view(b, -1, h * out.size(-1))
        )  # shape: (b, 1, h * d)
        out = out.squeeze(1)  # shape: (b, h * d)
        out = self.to_out(out)  # shape: (b, d)

        return out
