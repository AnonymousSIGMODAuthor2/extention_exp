from typing import Dict, List, Tuple
from models import Place


def HPF(pi: Place, pj: Place, W: float, psS, sS, k: int) -> float:
    K = W * k
    #
    return (K - k) * (pi.rF - pj.rF) / (k - 1) + (psS[pi.id] + psS[pj.id]) / (k - 1) - 2 * W * sS[(pi.id, pj.id)]

#######################################################################################################################
#######################################################################################################################
def HPFR(R: List[Place], baseline_psS: Dict[int, float], baseline_sS: Dict[Tuple[int, int], float], W: float, K):
    
    psR = {p.id: 0.0 for p in R}
    k = len(R)
    
    #psR
    for pi in R:
        for pj in R:
            if pi.id != pj.id:
                psR[pi.id] += spacial_proximity(baseline_sS, pi, pj)
                
    #
    score = 0
    score = (sum(p.rF for p in R))/(2*k) + sum(baseline_psS[p.id] - W * psR[p.id] for p in R)/(2 * k * (K-W))
    #score = (K - k) * (sum(p.rF for p in R)) + sum(baseline_psS[p.id] - W * psR[p.id] for p in R)

    
    return score, sum(baseline_psS[p.id] for p in R), sum(W*psR[p.id] for p in R)

def HPFR_div(R: List[Place], baseline_psS: Dict[int, float], baseline_sS: Dict[Tuple[int, int], float], W: float, K):
    
    psR = {p.id: 0.0 for p in R}
    k = len(R)
    
    #psR
    for pi in R:
        for pj in R:
            if pi.id != pj.id:
                psR[pi.id] += spacial_proximity(baseline_sS, pi, pj)
    
    score_rf = 0
    score_ps = 0
    score_rf = (sum(p.rF for p in R))
    score_ps = sum(baseline_psS[p.id] - W * psR[p.id] for p in R)
    
    return score_rf, score_ps, sum(baseline_psS[p.id] for p in R), sum(W*psR[p.id] for p in R)

# use symmetric sS
def spacial_proximity(sS, pi, pj):
    return sS.get((pi.id, pj.id)) or sS.get((pj.id, pi.id)) or 0.0