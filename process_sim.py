"""
作者：Zby

日期：2024年01月12日

"""

import numpy as np

#According to the literature, select drugs with scores greater than 0.4 to determine their similarity




def create_sim():

    drug_sim_path = 'data/Similarity_Matrix_Drugs.txt'

    sim_drug_drug = np.loadtxt(drug_sim_path)

    sim_drug_drug[sim_drug_drug >= 0.4] = 1

    sim_drug_drug[sim_drug_drug < 0.4] 	= 0

    return(sim_drug_drug)