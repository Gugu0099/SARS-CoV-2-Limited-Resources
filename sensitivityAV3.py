from parameterSensitivity import *
import concurrent.futures
import csv
import numpy as np
from scipy.integrate import solve_ivp

def Diff(t1, y1, params):
    (
        k_TMP, k_Rel, k_UC, k_Rib_on, k_Rib_Prime, k_Rib_Term, k_Cleav,
        h_pp1, k_RdRp_on, k_RdRp_Prime, k_RdRp_Term, p, k_RdRp_Term_sg,
        h_nRNA, h_RdRp, k_NCap, lamda, b, k_Rib_Term_SEM, k_Rib_Term_N,
        h_SP, k_Cyto_ER, k_ER_ERGIC, c, k_Bud, d, k_Exo
    ) = params

    (
        SCoV2_ACE2, endoSCoV2, cgRNA, Rib_gRNA, T_RibgRNA, pp1, RdRp_gRNA, 
        T_RdRpgRNA, nRNA, RdRp_nRNA, T_RdRpnRNA, RdRp, gRNA, sgRNA, Rib_sgRNA, 
        T_RibsgRNA, N, Rib, sem_cyto, sem_ER, sem_ERGIC, NCap, Vir, SCoV2
    ) = y1

    d_SCoV2_ACE2 = -k_TMP * SCoV2_ACE2
    dendoSCoV2 = k_TMP * SCoV2_ACE2 - k_Rel * endoSCoV2
    dcgRNA = a * d * k_Rel * endoSCoV2 - k_UC * cgRNA
    dRib_gRNA = k_Rib_on * Rib * gRNA - k_Rib_Prime * Rib_gRNA
    dT_RibgRNA = k_Rib_Prime * Rib_gRNA - k_Rib_Term * T_RibgRNA
    dpp1 = k_Rib_Term * T_RibgRNA - (k_Cleav + np.log(2) / h_pp1) * pp1
    dRdRp_gRNA = k_RdRp_on * RdRp * gRNA - k_RdRp_Prime * RdRp_gRNA
    dT_RdRpgRNA = k_RdRp_Prime * RdRp_gRNA - k_RdRp_Term * T_RdRpgRNA
    dnRNA = (k_RdRp_Term * T_RdRpgRNA) - (k_RdRp_on * RdRp + np.log(2) / h_nRNA) * nRNA + k_RdRp_Prime * RdRp_nRNA
    dRdRp_nRNA = k_RdRp_on * RdRp * nRNA - k_RdRp_Prime * RdRp_nRNA
    dT_RdRpnRNA = k_RdRp_Prime * RdRp_nRNA - p * k_RdRp_Term * T_RdRpnRNA - (1 - p) * k_RdRp_Term_sg * T_RdRpnRNA
    dRdRp = k_Cleav * pp1 - (k_RdRp_on * gRNA + k_RdRp_on * nRNA + np.log(2) / h_RdRp) * RdRp + k_RdRp_Term * T_RdRpgRNA + p * k_RdRp_Term * T_RdRpnRNA + (1 - p) * k_RdRp_Term_sg * T_RdRpnRNA
    dgRNA = k_UC * cgRNA - (k_Rib_on * Rib + a * k_NCap * N + np.log(2) / h_gRNA) * gRNA + k_Rib_Prime * Rib_gRNA + k_RdRp_Prime * RdRp_gRNA + p * k_RdRp_Term * T_RdRpnRNA
    dsgRNA = (1 - p) * k_RdRp_Term_sg * T_RdRpnRNA - (k_Rib_on * Rib + np.log(2) / h_gRNA) * sgRNA + k_Rib_Prime * Rib_sgRNA
    dRib_sgRNA = k_Rib_on * Rib * sgRNA - k_Rib_Prime * Rib_sgRNA
    dT_RibsgRNA = k_Rib_Prime * Rib_sgRNA - lamda * k_Rib_Term_SEM * T_RibsgRNA - (1 - lamda) * k_Rib_Term_N * T_RibsgRNA
    dN = (1 - lamda) * k_Rib_Term_N * T_RibsgRNA - b * k_NCap * N * gRNA - np.log(2) / h_SP * N
    dRib = -k_Rib_on * Rib * (gRNA + sgRNA) + k_Rib_Term * T_RibgRNA + lamda * k_Rib_Term_SEM * T_RibsgRNA + (1 - lamda) * k_Rib_Term_N * T_RibsgRNA
    dSEM_cyto = lamda * k_Rib_Term_SEM * T_RibsgRNA - (k_Cyto_ER + np.log(2) / h_SP) * sem_cyto
    dSEM_ER = k_Cyto_ER * sem_cyto - k_ER_ERGIC * sem_ER
    dSEM_ERGIC = k_ER_ERGIC * sem_ER - c * k_Bud * sem_ERGIC * NCap
    dNCap = k_NCap * N * gRNA - d * k_Bud * sem_ERGIC * NCap
    dVir = k_Bud * sem_ERGIC * NCap - k_Exo * Vir
    dSCoV2 = k_Exo * Vir

    return [
        d_SCoV2_ACE2, dendoSCoV2, dcgRNA, dRib_gRNA, dT_RibgRNA, dpp1,
        dRdRp_gRNA, dT_RdRpgRNA, dnRNA, dRdRp_nRNA, dT_RdRpnRNA, dRdRp,
        dgRNA, dsgRNA, dRib_sgRNA, dT_RibsgRNA, dN, dRib, dSEM_cyto,
        dSEM_ER, dSEM_ERGIC, dNCap, dVir, dSCoV2 / 1e-6
    ]

def event(t1, y1, params):
    return y1[23] - 1000

event.terminal = True
event.direction = 1

y1_initial = np.array([
    SCoV2_Ace2_0, endoSCoV2_0, cgRNA_0, Rib_gRNA_0, T_RibgRNA_0, pp1_0,
    RdRp_gRNA_0, T_RdRpgRNA_0, nRNA_0, RdRp_nRNA_0, T_RdRpnRNA_0, RdRp_0,
    gRNA_0, sgRNA_0, Rib_sgRNA_0, T_RibsgRNA_0, N_0, Rib_0, sem_cyto_0,
    sem_ER_0, sem_ERGIC_0, NCap_0, Vir_0, SCoV2_0
])

def update_params(params_dict, param_name, param_value):
    params_dict[param_name] = param_value

    if param_name == 'k_Rib_Step':
        params_dict.update({
            'k_Rib_Prime': params_dict['k_Rib_Step'] / l_Rib_Primer,
            'k_Rib_Term': params_dict['k_Rib_Step'] / l_pp1,
            'k_Rib_Term_SEM': params_dict['k_Rib_Step'] / l_SEM,
            'k_Rib_Term_N': params_dict['k_Rib_Step'] / l_N
        })
    elif param_name == 'k_RdRp_Step':
        params_dict.update({
            'k_RdRp_Prime': params_dict['k_RdRp_Step'] / l_RdRp_Primer,
            'k_RdRp_Term': params_dict['k_RdRp_Step'] / l_gRNA,
            'k_RdRp_Term_sg': params_dict['k_RdRp_Step'] / l_sgRNA
        })

def run_sensitivity_analysis(change, param_name, param_base_value, y1_initial):
    results = []
    params_dict = globals().copy()

    for fold_change in change:
        param_value = param_base_value * fold_change
        update_params(params_dict, param_name, param_value)
        
        params = [
            params_dict[key] for key in [
                'k_TMP', 'k_Rel', 'k_UC', 'k_Rib_on', 'k_Rib_Prime', 'k_Rib_Term', 
                'k_Cleav', 'h_pp1', 'k_RdRp_on', 'k_RdRp_Prime', 'k_RdRp_Term', 'p', 
                'k_RdRp_Term_sg', 'h_nRNA', 'h_RdRp', 'k_NCap', 'lamda', 'b', 
                'k_Rib_Term_SEM', 'k_Rib_Term_N', 'h_SP', 'k_Cyto_ER', 'k_ER_ERGIC', 
                'c', 'k_Bud', 'd', 'k_Exo'
            ]
        ]

        t1 = (0, 48 * 3600)
        max_step = 1  # Adjust this value as needed

        # Solve the differential equations with 'DOP853' solver
        sol = solve_ivp(Diff, t1, y1_initial, args=(params,), events=event, method='DOP853', max_step=max_step, rtol=1e-6, atol=1e-8)
        time_result = sol.t[-1] / 3600
        results.append((param_name, fold_change, time_result))

    return results

if __name__ == '__main__':
    change = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

    parameters = {
        "k_TMP": 0.0043,
        "k_Rel": 0.005,
        "k_UC": 0.005,
        "k_Rib_on": 0.5,
        "k_Rib_Step": 6,
        "k_Cleav": 0.021,
        "h_pp1": 1 * 3600,
        "k_RdRp_on": 0.09,
        "h_RdRp": 2 * 3600,
        "k_RdRp_Step": 20,
        "k_NCap": 0.00001,
        "k_Bud": 0.01,
        "h_SP": 30 * 60,
        "k_Cyto_ER": 0.002,
        "k_ER_ERGIC": 0.002,
        "k_Exo": 0.0002,
    }

    with open('sensitivity_analysis_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Parameter', 'Fold Change', 'Time (hours)'])

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(run_sensitivity_analysis, change, param, value, y1_initial): param
                for param, value in parameters.items()
            }
            for future in concurrent.futures.as_completed(futures):
                results = future.result()
                for result in results:
                    writer.writerow(result)
                    csvfile.flush()