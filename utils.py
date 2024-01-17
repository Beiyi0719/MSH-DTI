"""
作者：Zby

日期：2023年06月06日

"""
import scipy.sparse as sp
import numpy as np
import torch

def _gcn(norm_adj, init_emb,num):
    #graph convolution

    ego_emb = init_emb
    all_emb = [ego_emb]
    for k in range(num):
        ego_emb = torch.sparse.mm(norm_adj, ego_emb)
        all_emb += [ego_emb]

    all_emb = torch.stack(all_emb, dim=1)
    all_emb = torch.mean(all_emb, dim=1)
    return all_emb

def _create_recsys_adj_mat(matrix, num1, num2):

    #create adjacency matrix
    matrix_adj = sp.csr.csr_matrix(matrix)
    drug_protein_idx = [[u, i] for (u, i), r in matrix_adj.todok().items()]
    drug_list, protein_list = list(zip(*drug_protein_idx))

    drug_np = np.array(drug_list, dtype=np.int32)
    protein_np = np.array(protein_list, dtype=np.int32)
    ratings = np.ones_like(drug_np, dtype=np.float32)
    n_nodes = num1 + num2
    tmp_adj = sp.csr_matrix((ratings, (drug_np, protein_np + num1)), shape=(n_nodes, n_nodes))  # (m+n)*(m+n)
    adj_mat = tmp_adj + tmp_adj.T

    return _normalize_spmat(adj_mat)


def _normalize_spmat(adj_mat):
    rowsum = np.array(adj_mat.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj_tmp = d_mat_inv.dot(adj_mat)
    adj_matrix = norm_adj_tmp.dot(d_mat_inv)
    return adj_matrix

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    indices = np.vstack((coo.row, coo.col)).astype(np.int64)
    values = coo.data.astype(np.float32)
    shape = coo.shape
    sparse_coo_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices),
                                                torch.FloatTensor(values),
                                                torch.Size(shape))
    #Convert Sparse Matrix
    return sparse_coo_tensor

def _create_relation_adj_mat(relation_matrix):
    relation_adj = sp.csr_matrix(relation_matrix)
    pp_idx = [[ui, uj] for (ui, uj), r in relation_adj.todok().items()]
    p1_idx1, p2_idx1 = list(zip(*pp_idx))
    ratings = np.ones_like(p1_idx1, dtype=np.float32)

    tmp_adj = sp.csr_matrix((ratings, (p1_idx1, p2_idx1)),
                            shape=(relation_matrix.shape[0], relation_matrix.shape[1]))
    adj_mat = tmp_adj + tmp_adj.T
    return _normalize_spmat(adj_mat)


