# Personal-project transfer audit and an independent FLARE design

Reviewed 2026-11-02. This is a code/document audit, not reproduction of either recommendation study. No training, personal-project experiment, or FLARE sealed/test-array read was performed. Both repositories were readable through the existing Git HTTPS authentication. Temporary checkouts are `/tmp/flare-uq-UncertaintyRec` and `/tmp/flare-uq-TGLRec`.

## 1. Versions and what these repositories actually implement

| Repository | Inspected HEAD | Current method |
|---|---|---|
| `appleweiping/UncertaintyRec` | `1c5f7cd366979d3c04402227c211b8b35920ee97` | PEAR v4: forecast-matched popularity drift, request-level drift factorization, deployment refresh |
| `appleweiping/TGLRec` | `99a356528476978bb9d4295b2dcce411a3c3cbc1` | TGLRec q3: temporal-neighbor proposals, exact catalog-normalized LLM candidate scores, evidence-gated mixture |

The names and old logs can mislead. UncertaintyRec's current method is not an uncertainty-calibrated reranker or the abandoned teacher-projection infrastructure in `src/`. Its `research/ACTIVE_ROUTE.md` explicitly identifies PEAR v4, Qwen3-8B, and deployment strength 1 as the current method. TGLRec's `DIRECTION_DECISION.md` freezes q3 with `rho=0`; two-sided gates, similarity-prior redesigns, and larger-backbone alternatives are historical. Those restrictions concern these donor projects; this audit proposes a separate FLARE study without modifying or restarting them.

### PEAR: implemented mechanism

1. A recent prior is a smoothed exponentially decayed count distribution. At reference time r, `c_h(i)=sum_{events before r} 2^{-(r-t)/h}` and `pi_h(i) proportional to c_h(i)+0.1*mean(c_h)`. The half-life is selected by validation-target log likelihood, not test ranking. `urec/deploy_prior.py:23`, `:43`, `:271` implement these operations.
2. Drift compares recent and long-run priors, `delta_r(i)=log pi_h,r(i)-log pi_inf,r(i)`. Current training uses start-of-month snapshots with strict `< month_start` inclusion. The helper's default is historical month-end, so calling defaults is not sufficient to reproduce the current method: `urec/scripts/run_pear4.sh:15` explicitly passes `--poe_snap start`, and `urec/deploy_prior.py:180` implements the branch.
3. `DriftGate` in `urec/common.py:219` computes `kappa(x)=2 sigmoid(w^T RMS(h_x)+b)`, initialized at 1. The prompt hidden state is detached in `urec/train_llm.py:511`. The active method does not use the optional staleness parameter found in the general class.
4. `urec/beam.py:473` and `:499-517` compute recent/long-run child mass ratios and insert gated drift **before each local trie softmax**. Thus complete-item log probability contains all local normalization terms. It is not generally identical to globally renormalizing `p_base(i)*exp(kappa*delta(i))` over items. The README states the correction more carefully than an older helper docstring.
5. Deployment refresh uses newly observed interactions before the deployment boundary without retraining the backbone; `run_pear4.sh:19-22` uses t80 for validation and t90 for test. Current `README.md:107` explicitly characterizes gate/confidence correlation as learned reliance, not a calibrated-uncertainty guarantee.

Source example: https://github.com/appleweiping/UncertaintyRec/blob/1c5f7cd366979d3c04402227c211b8b35920ee97/urec/common.py#L219

### TGLRec: implemented mechanism

1. `code/tglrec/tne.py:25-68` builds an expert from training histories, selects neighbors by shared history and temporal weights, then votes for candidates around an anchor with directional and distance decay. Only training histories populate the index; this is a recommendation co-consumption construction, not an astronomical light-curve likelihood.
2. `tne.py:84-101` returns a normalized item distribution and `E=sum neighbor cosine similarities`; it also reports `n_eff=(sum similarities)^2/sum similarities^2`. E is support for this expert, not a posterior probability of being correct.
3. `code/tglrec/verify.py:95-145` evaluates exact LLM title probabilities under trie-valid-token normalization. `:149-155` unions beam candidates and expert proposals. `:158-168` ranks by `(1-pi)*p_LLM+pi*p_T` in log space.
4. `code/tglrec/select_gate.py:66-67`, `:96-105` implements the frozen `dirichlet` form: fix the log-E coefficient at 1, select an intercept on validation, and obtain `pi=E/(E+alpha)` with alpha=exp(-intercept). E=0 gives pi=0. `evaluate.py:234-243` uses this gate at inference.
5. The default selector optimizes NDCG, not proper probabilistic calibration. The optional candidate-conditional likelihood in `select_gate.py:58-59` renormalizes only over candidates and cannot establish full-catalog probability calibration. `fit_calib.py` implements an older temperature/gate-bias option; its presence does not make it part of the frozen q3 method.
6. Remaining-catalog bounds in `decode.py:344-380` concern search completeness/top-K certification. They do not certify that a recommendation is correct or that its uncertainty is calibrated.

Source example: https://github.com/appleweiping/TGLRec/blob/99a356528476978bb9d4295b2dcce411a3c3cbc1/code/tglrec/tne.py#L84

## 2. Transfer map, including reasons not to transfer literally

| Donor mechanism | Useful FLARE translation | Constraint |
|---|---|---|
| PEAR: historical-availability discipline | All early features, neighbor reference sets, calibration versions, and any rates must use only information available at that horizon and alert date. | Object age (5/10/20 days since first detection) is not calendar-time drift. These axes need distinct experiments. |
| PEAR: refresh statistics instead of retraining a large model | Refresh a post-hoc calibration layer on verified new labels while preserving the classifier; record versioned calibration provenance. | Spectroscopic labels are selectively acquired. New labelled-class frequencies are not population priors unless the selection mechanism is addressed. |
| PEAR: bounded, learned reliance | Train a small score adapter or gating rule on honest development data, then freeze it before conformal calibration. | A bounded coefficient limits its numerical effect; it does not prove no harm or coverage. |
| TGLRec: shared evidence supports both expert and gate | Build a local, same-horizon reference expert; measure support and disagreement to decide how much its prediction contributes. | A dense neighborhood can be confidently wrong under covariate shift; evaluate against a constant-mixture and an expert-only control. |
| TGLRec: exact normalization and common candidate universe | All six FLARE classes stay present in every probability vector; calibrate the exact model/adapter used in deployment. | There is little candidate-search benefit with only six classes. LLM beam expansion is not a meaningful standalone astronomy contribution. |
| Both: explicit controls sharing the same information | Compare against temperature scaling, same-neighbor expert alone, fixed mixture, permuted support, and a size-matched lower-alpha baseline. | Adding data or increasing set size can explain an apparent improvement without a better UQ mechanism. |

Three non-transfers are especially important:

- A recommendation confidence score, E, kappa, or mixture weight is not a conformal p-value, an e-value, or an estimate of astronomical novelty.
- A popularity correction does not solve open-set recognition. Class-prior shift correction assumes known classes and restrictions on `P(X|Y)`; FLARE's sparse/missing photometry changes the measurement process, and truly novel objects have no known class prior to correct.
- Exponential weighting of calibration scores is not automatically distribution-free conformal prediction. Genuine weighted-conformal guarantees need the appropriate likelihood-ratio or other explicit assumptions, and estimated weights need their uncertainty disclosed.

## 3. Independent candidate design: support-adaptive score calibration

Working description, not a novelty claim: **a support-adaptive probability adapter followed by untouched split-conformal calibration**. This is a challenger to, not a replacement for, the sequential/e-value direction until a design comparison is completed.

### Research question

Can local reference support and model/reference disagreement improve the *efficiency* of correctly calibrated early-time FLARE sets and the ranking of costly review requests, beyond constant ensembling and simply lowering alpha? The motivation transfers TGLRec's evidence gate and PEAR's frozen-backbone refresh, while avoiding unsupported claims about their uncertainty semantics.

### A concrete three-stage algorithm

**A. Train/reference stage.** At each horizon h in {5,10,20,100}, fit or reuse a classifier only on a proper training split A. Build a reference expert from objects in A using the same causal horizon. One tractable representation is the fitted LightGBM trees' leaf-membership embedding; alternatively use prespecified robust-scaled physical features with an explicit missing-value mask. Choose the representation in the design, not after outer validation. Never put calibration or evaluation labels into the reference index. All horizons/views of an object belong to the same split.

For nonnegative neighborhood weights `w_j(x)` over the reference objects,

`p_N,c(x) = [sum_j w_j(x) 1(Y_j=c) + eta*pi_A,c] / [sum_j w_j(x)+eta]`.

Store support statistics `E=sum w`, `n_eff=(sum w)^2/sum(w^2)`, available-feature fraction, and disagreement between p_N and p_F. Choose a fixed support definition rather than labeling any convenient predictor "uncertainty".

**B. Honest score adaptation.** On disjoint gate-fitting objects B, select a single alpha>=0 for `lambda(x)=E/(E+alpha)` and optional single temperature T by multiclass log loss:

`q_c(x) = (1-lambda(x))*softmax(log p_F(x)/T)_c + lambda(x)*p_N,c(x)`.

Zero support forces lambda=0. Parameters/reference rules/model are then frozen. For an intentionally minimal first experiment, use T=1 and the one-parameter gate; temperature-only and jointly fitted temperature are separate named controls. Inner cross-validation may choose alpha on B, but B never serves as the final calibration pool. NLL fitting is a proper scoring objective; it does not itself prove calibration.

**C. Valid calibration.** On untouched class-labelled calibration objects C, compute `s_h(x,c)=1-q_h,c(x)` with exactly the frozen model, reference index, gate, horizon, and feature pipeline used for new objects. For class c, define

`p_h,c(x) = [1 + sum_{i in C:Y_i=c} 1{s_h(X_i,c)>=s_h(x,c)}] / (n_c+1)`.

Set `Gamma_h(x)={c:p_h,c(x)>alpha_h}`. With class-conditional exchangeability and no fit/calibration overlap, the standard rank proof gives `P(Y notin Gamma_h(X) | Y=c)<=alpha_h`; no local-conditional coverage follows. Rare classes may force very coarse p-values or full inclusion; report that limitation.

For a *known-vs-unknown* alarm under the composite known-class null, use `p_known,h=max_c p_h,c` rather than the p-value of the predicted class. Under any known true class c, `p_known,h>=p_h,c`, giving a valid p-value under the union null. This corresponds to an empty conformal set. For an initial lifetime-safe comparison, choose fixed nonnegative weights summing to one and alpha_h=alpha_life*w_h; a union bound controls any of the four scheduled alarms without independence. This is an honest baseline for the parent's stronger e-value study. It does not establish FDR across an adaptively selected or repeatedly tested stream.

### Development protocol and leakage controls

1. The existing BTS "test" has already informed the earlier study. Do not restore its untouched status by placing outputs in a directory named `sealed`. Call it descriptive replication only.
2. Use outer object-level development folds for screening. Within each outer training pool, keep A/B/C disjoint (or construct a valid resampling protocol explicitly). Class stratification must preserve sufficient C counts; report all small classes.
3. Scores of calibration and evaluation objects must come from the **same** frozen A model and reference/gate. A refit after calibration invalidates this construction. Calibration on OOF predictions combined with an unrelated full-data refit is excluded.
4. Duplicate context-dropout views are not independent calibration objects. Use one fixed random view per object for a declared view-mixture target, or an explicitly proved group/object procedure. Do not multiply calibration n by 16.
5. Labels revealed by follow-up cannot enter a reference/calibration refresh until their actual availability time. A future prospective Fritz evaluation needs the selection/verification mechanism and calibration population defined.
6. Full-curve classification, early-horizon classification, lifetime false alarms, and cross-object FDR are separate endpoints. No theorem for one is presented as a theorem for the others.

### Required controls

- Honest split-conformal class-LAC with unmodified FLARE, same A and C.
- Temperature-scaled FLARE with identical B budget and separate C.
- Local expert alone, identical A/index and C.
- Constant-mixture q, selected on B with same objective and grid budget.
- E/(E+alpha) mixture with support permuted across B/evaluation objects using a predeclared seed.
- Atypicality Mondrian and the previous branch's M6 under the same fit/calibration roles; these are strong related baselines, not new components.
- Size-matched base LAC, matched using B/development data only.
- For repeated alarms: naive 1%-per-look, Bonferroni/alpha-spending, the shipped lifetime calibration, and the selected e-value method if that design is adopted.

### Falsifiable hypotheses and decision rule

Before scoring, freeze the algorithm/grid/split seeds/endpoints. Suggested criterion: at nominal 90% class-conditional coverage, the support gate should reduce mean set size or improve singleton yield over both the strongest constant-mixture and temperature-only control, without increasing known-object lifetime false alarms. Estimate paired object-level intervals; include fit/calibration variability across outer folds or repeated splits. Use a registered primary horizon (e.g. day 10) and report the others without choosing the successful horizon after inspection.

Secondary: reduce Brier/NLL and selective-risk area; improve rare-class coverage diagnostics at fixed size; improve anomaly recall at matched known-object lifetime false-alarm budget; report latency/reference storage. A lower ECE alone is not success. A result that only beats the unmodified classifier but loses to constant mixing does not establish the support-gating mechanism. A result confined to energy-defined strata is insufficient to claim broad conditional calibration.

The 53 known anomaly examples are highly limited and repeatedly studied. Keep all families and report family-level counts. They cannot by themselves establish discovery performance on unseen families. No gain is assumed in advance.

### Novelty risk and priority

This candidate is implementable, interpretable, and directly linked to the user's experience, but nearest-neighbor mixtures and post-hoc conformal calibration are well studied. Its novelty burden is high: a mere transplanted gate plus standard split conformal is unlikely to support a top-tier methods claim. A credible contribution would need a new, justified mechanism for scarcity/measurement availability, a convincing cost/coverage advantage under correct splits, and prospective astronomy evidence.

My independent recommendation is to retain it as an **information-matched challenger or ablation**, and prioritize the deployment/calibration mismatch plus repeated-look decision problem if the literature audit finds a real gap there. The donor projects' strongest transferable skill is rigorous normalization, causal information accounting, and mechanism controls, not an obligation to insert an LLM or reuse their exact architecture.

## 4. Evidence caveats carried over from the donor projects

The current README files themselves disclose important limitations: PEAR's displayed eight-domain cohort was chosen after results, and same-information controls can exceed it (`UncertaintyRec/README.md:71`, `:115-124`). TGLRec's seven-domain main table was selected after inspecting nine domains, and stronger expert/similarity controls reduce its leadership scope (`TGLRec/README.md:78`, current delivery assessment). Those are useful lessons for FLARE: retain negative results and all predeclared horizons, compare equal information and cost, and do not inherit a broad "SOTA" claim from a project title or selected table.

No unpublished personal manuscript or result was sent to an external service during this audit. GitHub links above identify the user-authorized sources; future public FLARE documentation should summarize transferable ideas without copying private manuscript text or extensive unpublished result tables.
