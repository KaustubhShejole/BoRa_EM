# BoRa_EM

This repository contains the official implementation of the paper *"Finding the Signal in the Spam: Jointly Learning Rewards and Worker Reliability from Pairwise Comparisons,"* accepted at the Forty-Second Conference on Uncertainty in Artificial Intelligence (UAI 2026), Amsterdam, the Netherlands, August 17–21, 2026.

**Authors:** Kaustubh Shivshankar Shejole, Tanish Agarwal, Arpit Agarwal and Avishek Ghosh

(Kaustubh and Tanish contributed equally to this project.)

---

Expectation Maximization method for Boltzmann Rational Model learning both item rewards and annotator competencies jointly from pairwise preference data.

We refer BoRa_EM as PGEM (Polya-Gamma augmented Expectation Maximization) in code.

For BoRaEM (PGEM), the complete code is in pgem_initialize_beta_1.py.

| Method | Class / module | Notes |
|---|---|---|
| **BoRaEM (PGEM)** (ours) | `pgem_initialize_beta_1.py` → `PolyaGamma_EM` (used through `EMWrapper`) | Pólya–Gamma EM, joint item score + worker reliability estimation. |
| **Plain Bradley–Terry (BT)** | `opt_fair.py` → `PairwiseFctsGPU` / `opt_pairwise_gpu` | No worker model; adapted from [`choix`](https://github.com/lucasmaystre/choix), ported to GPU (`torch.optim.LBFGS` in place of `scipy`'s Newton-CG). |
| **CrowdBT** | `opt_fair.py` → `CrowdBT_3_0` | Per-worker scalar reliability in $[0,1]$ mixing the BT probability with its complement. Adapted/vectorized from [Bias-Aware-Ranker-from-Pairwise-comparisons](https://github.com/Ambress92/Bias-Aware-Ranker-from-Pairwise-comparisons). |
| **BARP** (Bias-Aware Ranker from Pairwise comparisons) | `opt_fair.py` → `BARP` | Models a per-worker *bias toward a known item class*, not generic reliability. Same source repo as CrowdBT above. |
| **Rank Centrality (RC)** | `opt_fair.py` → `RankCentrality` | Spectral/Markov-chain ranking from a pairwise-count matrix; GPU power-iteration for the stationary distribution. |
| **NoisyBT / FactorBT** | `opt_fair.py` → `NoisyBT_3_0` | Two-parameter worker model (skill *and* position/identity bias), adapted from [crowd-kit](https://github.com/Toloka/crowd-kit)'s `NoisyBradleyTerry`. |
| **HBTL** (Hetereogeneous BTL, gradient-EM) | `grad_em.py` → `GradientEMWrapper` / `GradientEM` | Same scaled-BT model as PGEM, but fit with first-order gradient descent (Adam) instead of Pólya–Gamma EM. |
| **HTCV** | `htcv_m.py` → `HTCVWrapper` / `HTCV` | Hetereogeneous Thurstone Case V Model: $P(i \succ j) = \Phi(\beta_k(r_i-r_j)/\sqrt2)$, fit by gradient descent on a numerically-stable log-CDF loss. |

`distribution_utils.py` provides the data-generating side: samplers for the CrowdBT
mixture model (`crowd_bt_dist`) and for the scaled-logit model PGEM itself assumes
(`logistic_preference_dist`, plus a memory-efficient rejection-free variant
`memory_efficient_sample`), used to build the synthetic experiments.