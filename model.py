"""
作者：Zby

日期：2023年06月06日

"""
import torch
import torch.nn as nn

import utils

import torch.nn.functional as F

class GTN(nn.Module):
    def __init__(self, A, drug_num, protein_num,drug_structure,protein_structure,args):

        super(GTN, self).__init__()
        self.drug_structure = drug_structure
        self.protein_structure = protein_structure
        self.input_size = protein_structure.size(-1)
        self.drug_num = drug_num
        self.protein_num = protein_num
        self.args = args

        self.Lin_drug    = torch.nn.Linear(160,128)
        self.Lin_protein = torch.nn.Linear(512,128)

        initializer = torch.nn.init.xavier_uniform_

        self.p_weight = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())
        self.d_weight_i = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())
        self.d_weight_s = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())

        self.pd_weight_p = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())
        self.pd_weight_d = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())
        self.dp_weight_p = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())
        self.dp_weight_d = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())

        self.WA_weight_protein = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())
        self.BA_weight_protein = nn.Parameter(torch.zeros(128).requires_grad_())
        self.HA_weight_protein = nn.Parameter(torch.ones(128, 1) * 0.01)

        self.WB_weight_protein = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())
        self.BB_weight_protein = nn.Parameter(torch.zeros(128).requires_grad_())
        self.HB_weight_protein = nn.Parameter(torch.ones(128, 1) * 0.01)

        self.WA_weight_drug = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())
        self.BA_weight_drug = nn.Parameter(torch.zeros(128).requires_grad_())
        self.HA_weight_drug = nn.Parameter(torch.ones(128, 1) * 0.01)

        self.WB_weight_drug = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())
        self.BB_weight_drug = nn.Parameter(torch.zeros(128).requires_grad_())
        self.HB_weight_drug = nn.Parameter(torch.ones(128, 1) * 0.01)

        self.WA_weight_sim = nn.Parameter(initializer(torch.Tensor(128, 128)).requires_grad_())
        self.BA_weight_sim = nn.Parameter(torch.zeros(128).requires_grad_())
        self.HA_weight_sim = nn.Parameter(torch.ones(128, 1) * 0.01)

        self.dropout = torch.nn.Dropout(p=0.1)
    def forward(self, A):

        relation_matrix = A[0, :self.drug_num, (self.drug_num):(self.drug_num + self.protein_num)]
        dru_str  = self.Lin_drug(self.drug_structure)
        pro_str  = self.Lin_protein(self.protein_structure)

        #heterogeneous aggragation module
        dru_rel_emb, dru_rel_nei = self.get_rel_emb(A,dru_str,pro_str,num=self.args.layer,name = 'drug')

        pro_rel_emb, pro_rel_nei = self.get_rel_emb(A,dru_str,pro_str,num=self.args.layer,name='protein')
        dru_sim_emb, dru_sim_nei = self.get_rel_emb(A, dru_str, pro_str,num=self.args.layer, name='sim')

        dru_nei_emb = dru_rel_nei + dru_sim_nei
        dru_tem_emb \
            = torch.stack([torch.mean(torch.stack([pro_rel_nei[j] for j in range(relation_matrix.shape[1]) if relation_matrix[i][j] == 1.0], dim=1), dim=1)
                           if any(relation_matrix[i] == 1.0) else torch.zeros(128) for i in range(relation_matrix.shape[0])],dim=0)
        pro_tem_emb \
            = torch.stack([torch.mean(torch.stack([dru_nei_emb[j] for j in range(relation_matrix.T.shape[1]) if relation_matrix.T[i][j] == 1.0], dim=1), dim=1)
                           if any(relation_matrix.T[i] == 1.0) else torch.zeros(128) for i in range(relation_matrix.T.shape[0])],dim=0)

        dru_int_emb,pro_int_emb \
            = self.get_int_emb(dru_str,pro_str,dru_tem_emb,pro_tem_emb,relation_matrix, num=self.args.layer)

        drug_w \
            = self.get_weight(dru_int_emb, self.WA_weight_drug, self.BA_weight_drug, self.HA_weight_drug)
        pro_w \
            = self.get_weight(pro_int_emb, self.WA_weight_protein, self.BA_weight_protein,
                                    self.HA_weight_protein)

        dru_rel_w  \
            = self.get_weight(dru_rel_emb, self.WB_weight_drug,
                                              self.BB_weight_drug,self.HB_weight_drug)
        pro_rel_w  \
            = self.get_weight(pro_rel_emb, self.WB_weight_protein,
                                             self.BB_weight_protein, self.HB_weight_protein)
        dru_sim_w  \
            = self.get_weight(dru_sim_emb,self.WA_weight_sim,self.BA_weight_sim,
                                     self.HA_weight_sim)
        fin_dru_emb \
            = self.get_final_dru_emb(dru_int_emb, dru_rel_emb,dru_sim_emb, drug_w,
                                                          dru_rel_w,dru_sim_w)
        fin_pro_emb \
            = self.get_final_pro_emb(pro_int_emb, pro_rel_emb,
                                                             pro_w, pro_rel_w)


        y = torch.matmul(fin_dru_emb,fin_pro_emb.T)
        y = (y - torch.mean(y)) / torch.std(y)

        y = torch.sigmoid(y)
        return y

    def get_int_emb(self,dru_str,pro_str,add_emb_one,add_emb_two,matrix,num):

        one_one_relation = utils._create_recsys_adj_mat(matrix,matrix.shape[0],matrix.shape[1])

        one_one_relation = utils._convert_sp_mat_to_sp_tensor(one_one_relation)

        one_emb = dru_str
        two_emb = pro_str

        one_all_emb = 0.8 * one_emb + 0.2 * add_emb_one
        two_all_emb = 0.8 * two_emb + 0.2 * add_emb_two

        one_emb = torch.matmul(one_emb,self.pd_weight_d)
        two_all_emb    = torch.matmul(two_all_emb,self.dp_weight_p)

        two_emb = torch.matmul(two_emb,self.pd_weight_p)
        two_all_emb   = torch.matmul(two_all_emb,self.pd_weight_d)

        all_emb_d = torch.cat([one_emb, two_all_emb], axis=0)
        all_emb_d = utils._gcn(one_one_relation, all_emb_d,num)
        one_emb_fin, two_emb_dis = torch.split(all_emb_d, [matrix.shape[0], matrix.shape[1]], 0)

        all_emb_p = torch.cat([one_all_emb, two_emb], axis=0)
        all_emb_p = utils._gcn(one_one_relation, all_emb_p, num)
        one_emb_dis, two_emb_fin = torch.split(all_emb_p, [matrix.shape[0], matrix.shape[1]], 0)

        #two_emb_dis and one_emb_dis are discarded


        return one_emb_fin,two_emb_fin

    def get_rel_emb(self,A,drug_structure,protein_structure,num,name = ''):
        if name == 'drug':
            matrix = A[2,:self.drug_num,:self.drug_num]
            feature_emb = drug_structure
            weight = self.d_weight_i

        elif name == 'protein':
            matrix = A[3,(self.drug_num):(self.drug_num + self.protein_num),(self.drug_num):(self.drug_num + self.protein_num)]
            feature_emb = protein_structure
            weight = self.p_weight

        else:
            matrix = A[4,:self.drug_num,:self.drug_num]
            feature_emb = drug_structure
            weight = self.d_weight_i

        one_one_relation = utils._create_relation_adj_mat(matrix)

        one_one_relation = utils._convert_sp_mat_to_sp_tensor(one_one_relation)

        one_neighbor_emb = [torch.mean(torch.stack([feature_emb[j] for j in range(matrix.shape[0]) if matrix[i][j] == 1.0], dim=1), dim=1) if any(matrix[i] == 1.0) else torch.zeros(128) for i in range(matrix.shape[0])]
        feature_emb = torch.matmul(feature_emb, weight)
        feature_relation_emb = utils._gcn(one_one_relation, feature_emb,num)

        return feature_relation_emb,one_neighbor_emb

    def get_weight(self,emb,WA,BA,HA):

        logits = torch.matmul(F.relu(torch.matmul(emb, WA) + BA), HA)
        weight_w = F.log_softmax(logits,dim=0)
        weight_w = weight_w
        return weight_w

    def get_final_dru_emb(self,a_emb,b_emb,c_emb,a_weight,b_weight,c_weight):
        a_wiehgt = a_weight / (a_weight + b_weight + c_weight)
        b_weight = b_weight / (a_wiehgt + b_weight + c_weight)
        c_weight = 1.0 - a_wiehgt - b_weight
        a_emb = self.dropout(a_emb)
        b_emb = self.dropout(b_emb)
        c_emb = self.dropout(c_emb)
        fin_emb = a_wiehgt * a_emb + b_weight * b_emb + c_weight * c_emb
        return fin_emb

    def get_final_pro_emb(self,a_emb,b_emb,a_wiehgt,b_weight):
        a_wiehgt = a_wiehgt / (a_wiehgt + b_weight)
        b_weight = 1.0 - a_wiehgt
        a_emb = self.dropout(a_emb)
        b_emb = self.dropout(b_emb)
        fin_emb = a_wiehgt * a_emb + b_weight * b_emb
        return fin_emb
