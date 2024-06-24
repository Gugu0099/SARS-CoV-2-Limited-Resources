import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import solve_ivp # package to integrate ODE
from parameterSensitivity import *

def Diff(t1, y1):
    [
        SCoV2_ACE2,
        endoSCoV2,
        cgRNA,
        Rib_gRNA,
        T_RibgRNA,
        pp1,
        RdRp_gRNA,
        T_RdRpgRNA,
        nRNA,
        RdRp_nRNA,
        T_RdRpnRNA,
        RdRp,
        gRNA,
        sgRNA,
        Rib_sgRNA,
        T_RibsgRNA,
        N,
        Rib,
        sem_cyto,
        sem_ER,
        sem_ERGIC,
        NCap,
        Vir,
        SCoV2,
     ] = y1
    # EQ.1
    d_SCoV2_ACE2 = (-k_TMP) * (SCoV2_ACE2)
    # EQ.2
    dendoSCoV2 = (k_TMP * (SCoV2_ACE2)) - (k_Rel * (endoSCoV2))
    # EQ.3
    dcgRNA = (a * d * k_Rel * (endoSCoV2)) - k_UC * (cgRNA)
    # Eq.4
    dRib_gRNA = k_Rib_on * (Rib) * (gRNA) - k_Rib_Prime * Rib_gRNA
    # Eq.5
    dT_RibgRNA = k_Rib_Prime * Rib_gRNA - k_Rib_Term * T_RibgRNA
    # Eq.8
    dpp1 = k_Rib_Term * T_RibgRNA - (k_Cleav + (math.log(2) / h_pp1)) * pp1
    # Eq.9
    dRdRp_gRNA = k_RdRp_on * RdRp * gRNA - k_RdRp_Prime * RdRp_gRNA
    # Eq.10
    dT_RdRpgRNA = k_RdRp_Prime * RdRp_gRNA - k_RdRp_Term * T_RdRpgRNA
    # Eq.13
    dnRNA = (
        (k_RdRp_Term * T_RdRpgRNA)
        - (k_RdRp_on * RdRp + (math.log(2) / h_nRNA)) * nRNA
        + k_RdRp_Prime * RdRp_nRNA
    )
    # Eq.14
    dRdRp_nRNA = k_RdRp_on * RdRp * nRNA - k_RdRp_Prime * RdRp_nRNA
    # Eq.15
    dT_RdRpnRNA = (
        k_RdRp_Prime * (RdRp_nRNA)
        - p * (k_RdRp_Term * T_RdRpnRNA)
        - (1 - p) * (k_RdRp_Term_sg * T_RdRpnRNA)
    )
    # Eq.16
    dRdRp = (
        k_Cleav * pp1
        - (k_RdRp_on * gRNA + k_RdRp_on * nRNA + (math.log(2) / h_RdRp)) * RdRp
        + (k_RdRp_Term * T_RdRpgRNA)
        + p * (k_RdRp_Term * T_RdRpnRNA)
        + (1 - p) * (k_RdRp_Term_sg * T_RdRpnRNA)
    )
    # Eq.18
    dgRNA = (
        k_UC * cgRNA
        - (k_Rib_on * Rib + a * (k_NCap * N) + (math.log(2) / h_gRNA)) * gRNA
        + (k_Rib_Prime * (Rib_gRNA))
        + (k_RdRp_Prime * (RdRp_gRNA))
        + p * (k_RdRp_Term * T_RdRpnRNA)
    )
    # Eq.19
    dsgRNA = (
        (1 - p) * (k_RdRp_Term_sg * T_RdRpnRNA)
        - (k_Rib_on * Rib + (math.log(2) / h_gRNA)) * sgRNA
        + k_Rib_Prime * Rib_sgRNA
    )
    # Eq.20
    dRib_sgRNA = k_Rib_on * Rib * sgRNA - k_Rib_Prime * (Rib_sgRNA)
    # Eq.21
    dT_RibsgRNA = (
        k_Rib_Prime * Rib_sgRNA
        - lamda * (k_Rib_Term_SEM * T_RibsgRNA)
        - (1 - lamda) * (k_Rib_Term_N * T_RibsgRNA)
    )
    # Eq.22
    dN = (
        (1 - lamda) * (k_Rib_Term_N * T_RibsgRNA)
        - (b * (k_NCap * N * gRNA))
        - (math.log(2) / h_SP) * N
    )
    # Eq.23
    dRib = (
        -k_Rib_on * Rib * (gRNA + sgRNA)
        + (k_Rib_Term * T_RibgRNA)
        + lamda * (k_Rib_Term_SEM * T_RibsgRNA)
        + (1 - lamda) * (k_Rib_Term_N * T_RibsgRNA)
    )
    # Eq.26
    dSEM_cyto = (
        lamda * (k_Rib_Term_SEM * T_RibsgRNA)
        - (k_Cyto_ER + (math.log(2) / h_SP)) * sem_cyto
    )
    # Eq.27
    dSEM_ER = k_Cyto_ER * sem_cyto - k_ER_ERGIC * sem_ER
    # Eq.28
    dSEM_ERGIC = k_ER_ERGIC * sem_ER - c * (k_Bud * sem_ERGIC * NCap)
    # Eq.29
    dNCap = k_NCap * N * gRNA - d * (k_Bud * sem_ERGIC * NCap)
    # Eq. 20
    dVir = k_Bud * sem_ERGIC * NCap - k_Exo * Vir
    # dy(1)
    dSCoV2 = k_Exo * Vir

    return [
        # 0
        d_SCoV2_ACE2,
        # 1
        dendoSCoV2,
        # 2
        dcgRNA,
        # 3
        dRib_gRNA,
        # 4
        dT_RibgRNA,
        # 5
        dpp1,
        # 6
        dRdRp_gRNA,
        # 7
        dT_RdRpgRNA,
        # 8
        dnRNA,
        # 9
        dRdRp_nRNA,
        # 10
        dT_RdRpnRNA,
        # 11
        dRdRp,
        # 12
        dgRNA,
        # 13
        dsgRNA,
        # 14
        dRib_sgRNA,
        # 15
        dT_RibsgRNA,
        # 16
        dN,
        # 17
        dRib,
        # 18
        dSEM_cyto,
        # 19
        dSEM_ER,
        # 20
        dSEM_ERGIC,
        # 21
        dNCap,
        # 22
        dVir,
        # 23
        dSCoV2/(10**-6),
    ]

def event(t1, y1):
    return y1[23] - 1000

event.terminal = True
event.direction = 1

y1 = [
    SCoV2_Ace2_0,
    endoSCoV2_0,
    cgRNA_0,
    Rib_gRNA_0,
    T_RibgRNA_0,
    pp1_0,
    RdRp_gRNA_0,
    T_RdRpgRNA_0,
    nRNA_0,
    RdRp_nRNA_0,
    T_RdRpnRNA_0,
    RdRp_0,
    gRNA_0,
    sgRNA_0,
    Rib_sgRNA_0,
    T_RibsgRNA_0,
    N_0,
    Rib_0,
    sem_cyto_0,
    sem_ER_0,
    sem_ERGIC_0,
    NCap_0,
    Vir_0,
    SCoV2_0,
]
# fold-change array
change = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
# parameters
# 1
# file
# f = open("Data.txt", "w")
# f.write("k_TMP: \n")
'''
print("k_TMP:")
for i in range(len(change)):
    k_TMP = 0.0043 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

# f.write("k_Rel: \n")  
print("k_Rel:") 
for i in range(len(change)):
    k_Rel = 0.005 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")  

print("k_UC:") 
# f.write("k_UC: \n")
for i in range(len(change)):
    k_UC = 0.005 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("k_Rib_on:") 
# f.write("k_Rib_on: \n")
for i in range(len(change)):
    k_Rib_on = 0.5 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("k_Rib_Step:") 
# f.write("k_Rib_Step: \n")
for i in range(len(change)):
    k_Rib_Step = 6 * change[i]
    k_Rib_Prime = k_Rib_Step / l_Rib_Primer
    k_Rib_Term = k_Rib_Step / l_pp1
    k_Rib_Term_SEM = k_Rib_Step / l_SEM
    k_Rib_Term_N = k_Rib_Step / l_N
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("k_Cleav:") 
# f.write("k_Cleav: \n")
for i in range(len(change)):
    k_Cleav = 0.021 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("h_pp1:") 
# f.write("h_pp1: \n")   
for i in range(len(change)):
    h_pp1 = 1 * 60 * 60 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("k_RdRp_on") 
# f.write("k_RdRp_on \n")
for i in range(len(change)):
    k_RdRp_on = 0.09 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("h_RdRp:") 
# f.write("h_RdRp: \n")
for i in range(len(change)):
    h_RdRp = 2 * 60 * 60 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("k_RdRp_Step:") 
# f.write("k_RdRp_Step: \n")
for i in range(len(change)):
    k_RdRp_Step = 20 * change[i]
    k_RdRp_Prime = k_RdRp_Step / l_RdRp_Primer
    k_RdRp_Term = k_RdRp_Step / l_gRNA
    k_RdRp_Term_sg = k_RdRp_Step / l_sgRNA
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("k_NCap:") 
# f.write("k_NCap: \n")
for i in range(len(change)):
    k_NCap = 0.00001 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("k_Bud:")  
# f.write("k_Bud: \n") 
for i in range(len(change)):
    k_Bud = 0.01 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("h_SP:") 
# f.write("h_SP: \n")    
for i in range(len(change)):
    h_SP = 30 * 60 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("k_Cyto_ER:") 
# f.write("k_Cyto_ER: \n")    
for i in range(len(change)):
    k_Cyto_ER = 0.002 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")

print("k_ER_ERGIC:") 
# f.write("k_ER_ERGIC: \n")
for i in range(len(change)):
    k_ER_ERGIC = 0.002 * change[i]
    t1 = (0, 30*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")
'''
print("k_Exo:") 
# f.write("k_Exo: \n")
for i in range(len(change)):
    k_Exo = 0.0002 * change[i]
    t1 = (0, 48*60*60)
    fig2 = solve_ivp(Diff, t1, y1, events=event)
    print(fig2.t[-1]/ 3600)
    # f.write(str(fig2.t[-1]/ 3600) + "\n")
print("done")