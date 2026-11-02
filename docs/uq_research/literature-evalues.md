# E-values, repeated looks, and discovery control: verified research audit

Prepared 2026-11-02. This is a literature/theory/design audit, not an experimental result. No sealed prediction arrays or test labels were opened. Primary full text is cached in `sources-evalues/` beside this file.

## Decision-relevant findings

1. Fold-specific conformal **e**-prediction is a real, established way to remove the fold-versus-refit mismatch while retaining a finite-sample guarantee. Merely averaging fold conformal **p**-values, or pooling scores from unlike models, does not give the same guarantee.
2. The proposed combination of fold averaging, fixed look weights, and e-BH is mathematically defensible if its inputs are genuine individual null e-values and the population assumptions are satisfied. It is not a new general theorem: each component is established, and the stopped fixed-budget argument is elementary domination.
3. The most relevant derandomization paper uses *generalized*, batch-valid e-values. Its individual null means may exceed one. Copying those statistics into a per-object anytime claim is invalid without a new argument.
4. Power is a feasibility constraint. Generic p-to-e conversion can make discovery mathematically impossible at the desired sparse-anomaly prevalence, before any model quality is considered.
5. A calibrated whole-trajectory maximum is a strong, simpler baseline. It uses dependence across the four looks instead of paying an equal evidence budget at each look. FLARE already contains a whole-life calibration layer, so the actual contribution would be independent calibration, model alignment, validation, and perhaps efficiency—not inventing lifetime significance.

## Verified primary literature

### A. Directly relevant foundations with verified publication venues

- **Bashari, Epstein, Romano & Sesia (NeurIPS 2023), “Derandomized novelty detection with FDR control via conformal e-values.”** [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cec8ad7715d0d13899d5d7d31970f527-Abstract-Conference.html), [full text](https://arxiv.org/html/2302.07294), arXiv v3 updated 2023-10-23. Read Sections 2, 3.1–3.4 and the power discussion. Eq. (5) is a batch-adaptive threshold e-value; Theorem 3.2 proves `sum_{null j} E[e_j] <= n_test`, not `E[e_j] <= 1` for every j. The text explicitly says individual means may exceed one. Assumption 3.1 requires reference inliers and null test points to be exchangeable conditional on non-null test points. Theorem 3.6 permits only carefully permutation-invariant adaptive weights. Appendix S5 and the main text warn that discrete conformal p-values make generic p-to-e conversion weak.
- **Wang & Ramdas (JRSS B 2022), “False discovery rate control with e-values.”** [DOI](https://doi.org/10.1111/rssb.12489), [full text](https://arxiv.org/html/2009.02824). Journal/DOI verified through the Crossref registry; publisher webpage returned 403, so the theorem was read from the authors’ arXiv full text. Theorem 2 and Proposition 2 prove e-BH FDR control under arbitrary dependence. For m hypotheses, sort e-values descending, find the largest r with `e_(r) >= m/(q r)`, and reject the top r. Each individual null must have mean <=1, or a separately proved aggregate null-mean bound must apply. This is a finite-family statement, not automatic control over an expanding survey stream.
- **Vovk (Pattern Recognition 166, 2025, article 111674), “Conformal e-prediction.”** [DOI](https://doi.org/10.1016/j.patcog.2025.111674), [full text](https://arxiv.org/html/2001.05989). Publication verified in arXiv journal metadata; the original preprint dates to 2020, v5 to 2025-05-18. Section 5, Proposition 10 proves split conformal e-prediction; Section 6, Proposition 11 proves arithmetic-mean cross-conformal e-prediction under exchangeability. Ordinary cross-conformal p-value validity is explicitly distinguished from this exact e-value result.
- **Xu & Ramdas (AISTATS 2024), “Online multiple testing with e-values.”** [Official PMLR paper](https://proceedings.mlr.press/v238/xu24a.html), [full text](https://arxiv.org/html/2311.06412). Introduces e-LOND for an indefinite stream of hypotheses under arbitrary dependence. This is the relevant extension if new astronomical objects continually enter a discovery queue; repeatedly running batch e-BH on overlapping/changing batches is not a substitute.
- **Grünwald, de Heide & Koolen (JRSS B 2024), “Safe testing.”** [DOI](https://doi.org/10.1093/jrsssb/qkae011), [full text](https://arxiv.org/html/1906.07801). Journal/DOI verified through Crossref; arXiv also states accepted as a JRSS B discussion paper. Optional continuation concerns safely constructed evidence, not arbitrary products of marginally valid e-values.

### B. Recent, directly relevant work; do not invent venue labels

- **Gauthier, Bach & Jordan (2025), “E-Values Expand the Scope of Conformal Prediction.”** [Full text](https://arxiv.org/html/2503.13050), v3 updated 2025-05-06. Sections 1.1 and 2 give batch anytime-valid conformal prediction. Theorem 2.5 multiplies normalized-score e-values because each fresh calibration/test batch is exchangeable **conditional on previous batches**. Reusing the same light curve and calibration objects at days 5/10/20/100 does not establish that assumption. This is a close novelty collision for generic “anytime conformal e-values.” No journal/conference status asserted here.
- **Alami, Zakharia & Ben Taieb (2026), “Set-Preserving Calibration from Conformal P-Values to E-Values.”** [Full text](https://arxiv.org/html/2606.03600). Section 2.1 explains why generic calibrators enlarge sets; Proposition 2.4 and Theorem 2.6 construct an n- and alpha-dependent positive calibrator. Their conditions include `alpha(n+1)>1` and noninteger `alpha(n+1)`; exactness also depends on the rank-distribution assumptions in their setup. This is a recent preprint, not a claimed accepted paper. It makes “we improve cross-conformal sets by converting p to e” insufficient as a novelty claim.
- **Lee, Popov & Ren (2025/2026), “Full-conformal novelty detection.”** [Full text](https://arxiv.org/html/2501.02703), v3 updated 2026-10-29. ArXiv metadata says forthcoming in JASA. Provides full-data novelty detection, model selection and shift extensions using conformal e-values. It is a necessary comparator when claiming efficient use of scarce reference data. Abstract/introduction and method scope verified; its full proofs were not audited here.
- **Bai & Jin (2026), “Conformal Selective Prediction with General Risk Control.”** [Full text](https://arxiv.org/html/2603.24704). SCoRE controls risk among trusted predictions using generalized risk-weighted e-values. It separates error control for trusted classifications from novelty discovery. Preprint status only; not a fully audited theorem in this note.
- **“Copula Conformal Prediction for Multi-step Time Series Forecasting.”** [Full text](https://arxiv.org/html/2212.03281). Sections 3–4 distinguish independent/exchangeable whole sequences from dependent time points within a sequence, and use two-stage calibration to exploit temporal dependence. Section 4.1 and Theorem 4.1 are the closest conceptual baseline for jointly calibrating a fixed trajectory. The paper explicitly compares with Bonferroni-based CF-RNN. Its application is forecasting, not astronomical novelty detection; transfer needs a separate score and null definition.

These are primary-source verifications, not a claim that the entire relevant literature has been exhausted. Search used the arXiv API with e-values/conformal/trajectory/derandomization queries, official NeurIPS/PMLR proceedings, and exact-title DOI registry verification. A wrong Crossref top match was discarded rather than cited.

## Exact audit of the proposed guarantee

Let i denote an object, h one of the predeclared horizons, and k a model fold. Each score function must be fitted without its own calibration objects or the evaluated object. Early stopping, score normalization, hyperparameter tuning and anomaly-combiner training all count as fitting.

For nonnegative scores `g(s)` fixed from the proper-training data, a genuine split conformal e-value is

`E_ikh = (n_k+1) g(s_ikh) / (sum_{j in cal_k} g(s_jkh) + g(s_ikh))`.

Use strictly positive g or define an all-zero denominator symmetrically. Under exchangeability of calibration and null evaluation scores conditional on proper training, symmetry makes the mean one (or <=1 for a conservative variant). The test score in the denominator is essential. A construction dividing by the calibration mean alone generally lacks this finite-sample guarantee.

The arithmetic fold average `E_ih = sum_k v_k E_ikh`, for fixed nonnegative weights summing to one, remains an e-value without fold independence. Weights learned on the same calibration/test evidence do not automatically retain validity. Independent proper-training tuning, a valid symmetry argument, or a separate proof is required.

For fixed nonnegative look weights `w_h` with `sum_h w_h <=1`, define

`S_i,t = sum_{h <= t} w_h E_ih`, and `Z_i = sum_all_h w_h E_ih`.

Then `0 <= S_i,t <= Z_i` pathwise and `E[Z_i] <=1`. Hence for **any** selected time tau, `E[S_i,tau] <=1`, and `P(sup_t S_i,t >=1/alpha) <=alpha`. No temporal independence is needed. The guarantee includes the randomness in the calibration sample; it is not a conditional-on-this-fixed-calibration-sample guarantee.

Terminology: `S_i,t` is generally not a supermartingale—it only increases. It is nevertheless an e-process in the broad stopped-expectation definition, via domination by a terminal integrable variable (or its conditional-expectation martingale). Do not call it an accumulating likelihood-ratio/test-martingale construction or multiply its marginal e-values without conditional validity. Defining the filtration and starting information is necessary if using formal e-process language.

For a **fixed** family of m objects, applying e-BH to these stopped values controls FDR under arbitrary dependence. This even permits joint data-dependent stopping if every stopped value retains the same pathwise bound. For a fixed family, monotone values also make e-BH rejection sets nested. A growing set of objects, data-dependent screening of which objects count in m, or repeated decisions on overlapping batches needs a separate online selection argument.

### The null must match the science

- Class c e-values concern `Y=c`; they can generate prediction sets by retaining labels whose evidence against c is small.
- A calibrated score over known objects concerns a draw from the **known reference population mixture**. It controls false novelty flags only if the deployment known population is exchangeable with that reference, including class mixture, selection, context missingness and cadence.
- “Known class, under any class prior” is a stronger union/composite null. A conservative construction is `min_c E_c`, because it is bounded by the true-class e-value. This can be almost powerless when rare classes have tiny calibration pools.
- Prediction-set coverage, classifier correctness, out-of-taxonomy discovery FDR and alerts per day are distinct endpoints. None implies the others.

## Analytic power screen before fitting candidates

For a deterministic split-conformal rank p-value, `p >= 1/(n_cal+1)`. A valid generic calibrator `e=0.5/sqrt(p)` has cap `0.5 sqrt(n_cal+1)`.

Illustration using the approximately 1,784 calibration objects of a five-fold split of the 8,922-object development pool (not a measured experiment): the cap is about 21.1. At `q=.10`, e-BH needs `r/m >=1/(q e_max)`, about 47.4%, before any rejection is possible. A batch with 53 true anomalies among roughly 1,628 objects cannot attain useful sparse discovery with this construction unless it also flags many known objects. Raising model discrimination cannot remove this cap.

Normalized-score e-values are bounded by `n_cal+1`, about 1,785 here. A terminal batch of m=1,628 at q=.1 requires at least `ceil(m/(q*1785))=10` rejections in the best case. With four equal look weights, first-look cap is about 446, requiring at least 37 simultaneous rejections at that look. Actual scores will generally be well below these caps. Nested folds have smaller calibration pools and worse limits.

For class-conditional calibration, TDE has around a dozen members per fold; its e cap is correspondingly around a dozen. Taking a minimum over all class e-values inherits this rare-class ceiling. Do not silently pool classes to obtain power while continuing to claim a guarantee under arbitrary class-prior shift.

Compute these caps from actual split counts for every planned setting, state them before looking at outcomes, and reject demonstrably infeasible methods. This is more informative than promising that weighting will fix power.

## Independent research design: calibrated trajectory evidence for economical follow-up

### Question

Can a deployment-aligned, whole-object uncertainty layer lower known-source lifetime false alarms while detecting novel objects earlier, without the large power loss of naive evidence spending? The contribution sought is an astronomy-specific protocol plus measured improvement over strong baselines; no new universal e-value theorem is claimed.

### Candidate arms fixed before new scoring

1. Existing FLARE per-analysis review/consensus alarm, explicitly an uncorrected repeated-look comparator.
2. Existing FLARE lifetime replay, evaluated on genuinely disjoint objects; report its empirical nature and reference reuse separately.
3. Bonferroni or prespecified alpha-spending over the four matched-model look p-values.
4. Genuine normalized-score e-values, fixed fold weights and fixed look budgets, both per-object threshold and fixed-batch e-BH.
5. **Trajectory-envelope calibration**: proper-training-fixed per-model, per-horizon score transforms; calibrate the full trajectory maximum `A_k,j=max_h z_k,j,h`. At time t compare `A_k,i,t=max_{h<=t} z_k,i,h` with calibration full maxima. Its p-value is no smaller than its valid terminal p-value, giving a fixed-schedule lifetime error bound. Use one split for the clean reference; fold p-merging must itself be valid (Bonferroni-min is a baseline), and e conversion must pass the analytic power screen.

Do not standardize energy using evaluation objects. Changing the energy scale across models is harmless if each model scores both its own calibration trajectories and evaluation trajectory and the entire transformation is identical. It is harmful if one folds' calibration energy is compared directly with a refit's test energy.

### Whole-trajectory assumptions and caveats

Objects, not detections, are exchangeability units. All four horizons must share object IDs and the same fold partition. Missing/insufficient looks use a predeclared mapping, e.g. contribute negative infinity to a maximum until available; calibration and evaluation use identical eventual rules. Time origin must be prefix causal. Adaptive spectroscopic follow-up or photometry acquisition can change the trajectory distribution. A retrospectively fixed 5/10/20/100-day study establishes no guarantee for arbitrary per-detection monitoring beyond that schedule. The current model's per-detection lifetime schedule is different and should not be conflated with four-look results.

### Data/provenance plan

Retain outer development evaluation objects and nested fitting/calibration disjointness. Every trajectory records object ID, role, fold, horizon, exact model hash, source split hash and transformation version. The old BTS test set was already read; any new use is descriptive only. Seal outputs by role, but recognize that a directory named sealed does not undo past exposure. Preregister the next development comparison, then request fresh Fritz or another external labelled population for confirmatory claims. A paper-level generalization claim cannot be manufactured from resampling the same repeatedly inspected 53 anomalies.

### Outcomes and falsification

Primary operational outcome: recall by day 20 at a prespecified 1% **object-lifetime** false-alarm target, paired at object level; report detection delays and alerts per 1,000 known sources. Also report realized lifetime false-alarm fraction with binomial uncertainty, rare-family counts, class/cadence/context slices, and calibration variability across splits. FDR is a separate batch experiment: fix population mixture and batch size, report expected FDP over independent simulated batches with uncertainty; one batch's FDP is not an FDR estimate.

The candidate fails if it cannot beat an equally calibrated trajectory-max/Bonferroni baseline at comparable false-alarm load, if improvement disappears when paired model calibration is enforced, or if gain requires choosing weights after looking at evaluation anomalies. A negative power result is useful; it prevents shipping a theoretically valid but operationally useless detector.

## Internal FLARE novelty collision

`flare/lifetime.py` already forms running minima, calibrates them at analysis indices, and converts a whole-life minimum into a reference false-alarm fraction. `scripts/export_lifetime_far.py` builds both stages from the same replay population and explicitly labels its realized rates as in-sample, mentioning a split-half paper check. Therefore a new method should be positioned as **independently calibrated, model-aligned fixed-schedule lifetime monitoring**. Claiming to introduce repeated-look control to FLARE from scratch would misrepresent the existing framework.

## What is still unproven

No empirical superiority, external-data validity, paper acceptance probability, or new theoretical novelty has been established in this audit. The most defensible next action is a leakage-audited pilot with the prespecified strong baselines, followed by an explicit go/no-go decision and fresh-population validation if the pilot warrants it.
