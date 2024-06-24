from parameterSensitivity import *
import csv
import numpy as np
from scipy.integrate import solve_ivp
import time
import multiprocessing

# Global parameters dictionary
parameters = {
    "k_TMP": 0.0043,
    "k_Rel": 0.005,
    "k_UC": 0.005,
    "k_Rib_on": 0.5,
    "k_Rib_Step": 6,
    "k_Cleav": 0.021,
    "h_pp1": 1 * 60 * 60,
    "k_RdRp_on": 0.09,
    "h_RdRp": 2 * 60 * 60,
    "k_RdRp_Step": 20,
    "k_NCap": 0.00001,
    "k_Bud": 0.01,
    "h_SP": 30 * 60,
    "k_Cyto_ER": 0.002,
    "k_ER_ERGIC": 0.002,
    "k_Exo": 0.0002
}

# Initialize these variables globally
k_Rib_Prime = k_Rib_Step / l_Rib_Primer
k_Rib_Term = k_Rib_Step / l_pp1
k_Rib_Term_SEM = k_Rib_Step / l_SEM
k_Rib_Term_N = k_Rib_Step / l_N 
k_RdRp_Prime = k_RdRp_Step / l_RdRp_Primer
k_RdRp_Term = k_RdRp_Step / l_gRNA
k_RdRp_Term_sg = k_RdRp_Step / l_sgRNA

def Diff(t1, y1):
    global k_TMP, k_Rel, k_UC, k_Rib_on, k_Rib_Step, k_Cleav, h_pp1, k_RdRp_on, h_RdRp, k_RdRp_Step, k_NCap, k_Bud, h_SP, k_Cyto_ER, k_ER_ERGIC, k_Exo
    global k_Rib_Prime, k_Rib_Term, k_Rib_Term_SEM, k_Rib_Term_N, k_RdRp_Prime, k_RdRp_Term, k_RdRp_Term_sg
    
    (
        SCoV2_ACE2, endoSCoV2, cgRNA, Rib_gRNA, T_RibgRNA, pp1, RdRp_gRNA, 
        T_RdRpgRNA, nRNA, RdRp_nRNA, T_RdRpnRNA, RdRp, gRNA, sgRNA, 
        Rib_sgRNA, T_RibsgRNA, N, Rib, sem_cyto, sem_ER, sem_ERGIC, NCap, 
        Vir, SCoV2
    ) = y1
    
    # Constants
    log2 = np.log(2)
    
    # Equations
    d_SCoV2_ACE2 = -k_TMP * SCoV2_ACE2
    dendoSCoV2 = k_TMP * SCoV2_ACE2 - k_Rel * endoSCoV2
    dcgRNA = a * d * k_Rel * endoSCoV2 - k_UC * cgRNA
    dRib_gRNA = k_Rib_on * Rib * gRNA - k_Rib_Prime * Rib_gRNA
    dT_RibgRNA = k_Rib_Prime * Rib_gRNA - k_Rib_Term * T_RibgRNA
    dpp1 = k_Rib_Term * T_RibgRNA - (k_Cleav + log2 / h_pp1) * pp1
    dRdRp_gRNA = k_RdRp_on * RdRp * gRNA - k_RdRp_Prime * RdRp_gRNA
    dT_RdRpgRNA = k_RdRp_Prime * RdRp_gRNA - k_RdRp_Term * T_RdRpgRNA
    dnRNA = k_RdRp_Term * T_RdRpgRNA - (k_RdRp_on * RdRp + log2 / h_nRNA) * nRNA + k_RdRp_Prime * RdRp_nRNA
    dRdRp_nRNA = k_RdRp_on * RdRp * nRNA - k_RdRp_Prime * RdRp_nRNA
    dT_RdRpnRNA = k_RdRp_Prime * RdRp_nRNA - p * k_RdRp_Term * T_RdRpnRNA - (1 - p) * k_RdRp_Term_sg * T_RdRpnRNA
    dRdRp = (k_Cleav * pp1 
             - (k_RdRp_on * (gRNA + nRNA) + log2 / h_RdRp) * RdRp 
             + k_RdRp_Term * T_RdRpgRNA 
             + p * k_RdRp_Term * T_RdRpnRNA 
             + (1 - p) * k_RdRp_Term_sg * T_RdRpnRNA)
    dgRNA = (k_UC * cgRNA 
             - (k_Rib_on * Rib + a * k_NCap * N + log2 / h_gRNA) * gRNA 
             + k_Rib_Prime * Rib_gRNA 
             + k_RdRp_Prime * RdRp_gRNA 
             + p * k_RdRp_Term * T_RdRpnRNA)
    dsgRNA = ((1 - p) * k_RdRp_Term_sg * T_RdRpnRNA 
              - (k_Rib_on * Rib + log2 / h_gRNA) * sgRNA 
              + k_Rib_Prime * Rib_sgRNA)
    dRib_sgRNA = k_Rib_on * Rib * sgRNA - k_Rib_Prime * Rib_sgRNA
    dT_RibsgRNA = (k_Rib_Prime * Rib_sgRNA 
                   - lamda * k_Rib_Term_SEM * T_RibsgRNA 
                   - (1 - lamda) * k_Rib_Term_N * T_RibsgRNA)
    dN = ((1 - lamda) * k_Rib_Term_N * T_RibsgRNA 
          - b * k_NCap * N * gRNA 
          - log2 / h_SP * N)
    dRib = (-k_Rib_on * Rib * (gRNA + sgRNA) 
            + k_Rib_Term * T_RibgRNA 
            + lamda * k_Rib_Term_SEM * T_RibsgRNA 
            + (1 - lamda) * k_Rib_Term_N * T_RibsgRNA)
    dSEM_cyto = (lamda * k_Rib_Term_SEM * T_RibsgRNA 
                 - (k_Cyto_ER + log2 / h_SP) * sem_cyto)
    dSEM_ER = k_Cyto_ER * sem_cyto - k_ER_ERGIC * sem_ER
    dSEM_ERGIC = k_ER_ERGIC * sem_ER - c * k_Bud * sem_ERGIC * NCap
    dNCap = k_NCap * N * gRNA - d * k_Bud * sem_ERGIC * NCap
    dVir = k_Bud * sem_ERGIC * NCap - k_Exo * Vir
    dSCoV2 = k_Exo * Vir

    return [
        d_SCoV2_ACE2, dendoSCoV2, dcgRNA, dRib_gRNA, dT_RibgRNA, dpp1, dRdRp_gRNA, 
        dT_RdRpgRNA, dnRNA, dRdRp_nRNA, dT_RdRpnRNA, dRdRp, dgRNA, dsgRNA, 
        dRib_sgRNA, dT_RibsgRNA, dN, dRib, dSEM_cyto, dSEM_ER, dSEM_ERGIC, dNCap, 
        dVir, dSCoV2 / 10**-6
    ]

def event(t1, y1):
    return y1[23] - 1000

event.terminal = True
event.direction = 1

# Initial conditions
y1 = [
    SCoV2_Ace2_0, endoSCoV2_0, cgRNA_0, Rib_gRNA_0, T_RibgRNA_0, pp1_0, RdRp_gRNA_0, 
    T_RdRpgRNA_0, nRNA_0, RdRp_nRNA_0, T_RdRpnRNA_0, RdRp_0, gRNA_0, sgRNA_0, 
    Rib_sgRNA_0, T_RibsgRNA_0, N_0, Rib_0, sem_cyto_0, sem_ER_0, sem_ERGIC_0, 
    NCap_0, Vir_0, SCoV2_0
]

# Fold-change array
change = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
time_span = (0, 48 * 60 * 60)

def solve_for_param(param, base_value, scale, time_span, y1, fieldnames, lock):
    start_time = time.time()
    try:
        globals()[param] = base_value * scale
        print(f"Testing {param} with fold change {scale}")  # Debug print statement
        if param == "k_Rib_Step":
            global k_Rib_Prime, k_Rib_Term, k_Rib_Term_SEM, k_Rib_Term_N
            k_Rib_Prime = k_Rib_Step / l_Rib_Primer
            k_Rib_Term = k_Rib_Step / l_pp1
            k_Rib_Term_SEM = k_Rib_Step / l_SEM
            k_Rib_Term_N = k_Rib_Step / l_N
        if param == "k_RdRp_Step":
            global k_RdRp_Prime, k_RdRp_Term, k_RdRp_Term_sg
            k_RdRp_Prime = k_RdRp_Step / l_RdRp_Primer
            k_RdRp_Term = k_RdRp_Step / l_gRNA
            k_RdRp_Term_sg = k_RdRp_Step / l_sgRNA

        print(f"Solving for {param} with scale {scale}")  # Debug print statement

        fig2 = solve_ivp(Diff, time_span, y1, events=event)
        did_not_complete = fig2.t[-1] >= 48 * 3600  # Check if the simulation reached 48 hours
        viral_cycle_time = fig2.t[-1] / 3600

        result = {
            'Parameter': param,
            'FoldChange': scale,
            'ViralCycleTime': viral_cycle_time,
            'DidNotComplete': did_not_complete
        }

        print(f"Result for {param} with scale {scale}: {viral_cycle_time} hours, Did not complete: {did_not_complete}")
    except Exception as e:
        result = {
            'Parameter': param,
            'FoldChange': scale,
            'ViralCycleTime': None,
            'DidNotComplete': None,
            'Error': str(e)
        }
        print(f"Error with {param} at scale {scale}: {e}")
    end_time = time.time()
    print(f"Time taken for {param} with scale {scale}: {end_time - start_time} seconds")

    with lock:
        with open('results.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)

def write_results_to_csv(results, fieldnames):
    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for result in results:
            writer.writerow(result)

if __name__ == '__main__':
    fieldnames = ['Parameter', 'FoldChange', 'ViralCycleTime', 'DidNotComplete', 'Error']
    
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    pool = multiprocessing.Pool()
    lock = multiprocessing.Manager().Lock()

    tasks = [(param, base_value, scale, time_span, y1, fieldnames, lock) for param, base_value in parameters.items() for scale in change]
    pool.starmap(solve_for_param, tasks)

    pool.close()
    pool.join()