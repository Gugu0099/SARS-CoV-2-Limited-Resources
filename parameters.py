import numpy as np
import math


# initial condition
SCoV2_0 = 2 * 10**-6
SCoV2_Ace2_0 = 2 * 10**-6
endoSCoV2_0 = 0
cgRNA_0 = 0
gRNA_0 = 0
Rib_0 = 5  # uM
N_0 = 0
Rib_gRNA_0 = 0
RdRp_gRNA_0 = 0
T_RdRpnRNA_0 = 0
T_RibgRNA_0 = 0
T_RdRpgRNA_0 = 0
pp1_0 = 0
RdRp_0 = 0
nRNA_0 = 0
RdRp_nRNA_0 = 0
RdRp_0 = 0
sgRNA_0 = 0
Rib_sgRNA_0 = 0
T_RibsgRNA_0 = 0
sem_cyto_0 = 0
sem_ER_0 = 0
sem_ERGIC_0 = 0
NCap_0 = 0
Vir_0 = 0

# parameters
k_TMP = 0.0043  # s^-1
k_Rel = 0.005  # s^-1
k_UC = 0.005  # s^-1
k_Rib_on = 0.5  # um^-1s^1
k_Rib_Step = 6  # aa/s
k_Cleav = 0.021  # s^-1
h_pp1 = 1 * 60 * 60  # s
k_RdRp_on = 0.09  # um^-1s^1
h_RdRp = 2 * 60 * 60  # s
k_RdRp_Step = 20  # nt/s
k_NCap = 0.00001  # um^-1s^1
k_Bud = 0.01  # um^-1s^1
h_SP = 30 * 60  # s
h_gRNA = 1 * 60 * 60  # s
h_nRNA = 5 * 60  # s
k_Cyto_ER = 0.002  # s^-1
k_ER_ERGIC = 0.002  # s^-1
k_Exo = 0.0002  # s^-1
p = 0.75
lamda = 0.7
a = 1
b = 500
c = 2320
d = 2
l_Rib_Primer = 3  # aa
l_RdRp_Primer = 20  # nt
l_gRNA = 30000  # nt
l_sgRNA = 1120  # nt
l_pp1 = 6600  # aa
l_SEM = 354  # aa
l_N = 420  # aa
k_Rib_Prime = k_Rib_Step / l_Rib_Primer
k_Rib_Term = k_Rib_Step / l_pp1
k_RdRp_Prime = k_RdRp_Step / l_RdRp_Primer
k_RdRp_Term = k_RdRp_Step / l_gRNA
k_RdRp_Term_sg = k_RdRp_Step / l_sgRNA
k_Rib_Term_SEM = k_Rib_Step / l_SEM
k_Rib_Term_N = k_Rib_Step / l_N

# Parameters for SA