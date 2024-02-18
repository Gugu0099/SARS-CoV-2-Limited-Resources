import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import odeint  # package to integrate ODE
from parameters import *


def FunctionOfTime(t):
    # print(len(t1))
    time = 2 / (1 + np.exp(t))
    print(time)
    return time


def Diff(y1, t1):
    (
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
    ) = y1
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
    dT_RdRpgRNA = k_RdRp_Prime * RdRp_gRNA - FunctionOfTime(t1) * T_RdRpgRNA
    # Eq.13
    dnRNA = (
        FunctionOfTime(t1) * T_RdRpgRNA
        - (k_RdRp_on * RdRp + (math.log(2) / h_nRNA)) * nRNA
        + k_RdRp_Prime * RdRp_nRNA
    )
    # Eq.14
    dRdRp_nRNA = k_RdRp_on * RdRp * nRNA - k_RdRp_Prime * RdRp_nRNA
    # Eq.15
    dT_RdRpnRNA = (
        k_RdRp_Prime * (RdRp_nRNA)
        - p * (FunctionOfTime(t1) * T_RdRpnRNA)
        - (1 - p) * (k_RdRp_Term_sg * T_RdRpnRNA)
    )
    # Eq.16
    dRdRp = (
        k_Cleav * pp1
        - (k_RdRp_on * gRNA + k_RdRp_on * nRNA + (math.log(2) / h_RdRp)) * RdRp
        + FunctionOfTime(t1) * T_RdRpgRNA
        + p * (FunctionOfTime(t1) * T_RdRpnRNA)
        + (1 - p) * (k_RdRp_Term_sg * T_RdRpnRNA)
    )
    # Eq.18
    dgRNA = (
        k_UC * cgRNA
        - (k_Rib_on * Rib + a * (k_NCap * N) + (math.log(2) / h_gRNA)) * gRNA
        + (k_Rib_Prime * (Rib_gRNA))
        + (k_RdRp_Prime * (RdRp_gRNA))
        + p * (FunctionOfTime(t1) * T_RdRpnRNA)
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
        dSCoV2,
    ]


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
t1 = np.linspace(0, 30 * 60 * 60, 30 * 60 * 60)
fig2 = odeint(Diff, y1, t1)


plt.figure()
plt.plot(t1 / 3600, fig2[:, 12], label="i")
plt.xlabel("Time (hr)")
plt.ylabel("Concentration (µM)")
plt.legend()
plt.title("genomic RNA")
plt.show()
