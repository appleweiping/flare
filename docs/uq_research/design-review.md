# Independent design judge and completeness review

2026-11-02. Assessment before this new pilot's outcome inspection. This review uses the recovered FLARE code and UQ report; `audit-existing.md`, `personal-transfer.md`, `literature-evalues.md`, `literature-conditional.md`, and `literature-online.md`; primary full texts named in those memos; and the frozen `docs/uq_anytime_protocol.md` reported committed at `ba9e48e` before model fitting. No sealed arrays or new candidate performance results were read by this judge.

## Decision

Implement **B as an honest baseline, A as a power/validity challenger**, with identical object roles and score information. Treat the result as an exploratory engineering/science benchmark. Do not rename B as a new conformal method, claim A introduces e-merging, or select a success story after reading outcomes.

The longer-term scientific direction is **C: a source-level risk-calibrated early-decision or follow-up policy**, with a small evidence-adaptive probability adapter as a controlled component. It has a better operational objective and stronger connection to the user's experience, but the present specification and available data do not yet justify a top-tier paper claim. It needs rigorous finite-policy risk calibration, strong same-information controls, and an untouched astronomical cohort.

The recommendation of B is **not** an endorsement of the root agent's novelty: direct prior art already contains its core algorithm and theorem. Its immediate value is a transparent comparator and correcting FLARE's calibration/inference protocol.

## Scoring of three candidates

Scores 1–5, higher is better for the first four columns. Final column is risk of a false positive scientific conclusion, where **5 is worst**. Scores are judgement aids, not measurements or publication probabilities; differences of one point are not statistically meaningful.

| Candidate | Theoretical validity once fully specified | Verified novelty | Feasible with recovered assets | Potential astronomy utility | False conclusion risk |
|---|---:|---:|---:|---:|---:|
| **A: fold-matched e-values, fixed look mixture, e-BH** | 4 | 1 | 3 | 2 | 4 |
| **B: split trajectory-envelope lifetime calibration** | 4 | 1 | 4 | 3 | 3 |
| **C: evidence-adaptive early-decision/selection policy with held-out risk calibration** | 2 at current specification; potentially 4 | 2, unestablished | 2–3 | 4 | 4 |

No numerical average is used: these methods partly control different events, and a useful policy cannot compensate for an invalid guarantee.

### A: why it is worth a benchmark, not a default research winner

Strengths: valid per-model e-values can be averaged under arbitrary fold dependence; fixed nonnegative horizon budgets give a terminal dominating e-value and hence a stopped/ever-crossing bound. Fixed-batch e-BH provides an established arbitrary-dependence FDR guarantee. It addresses the prior fold/refit mismatch in a principled way.

Critical objections:

- The nearest derandomization work uses generalized batch-valid e-values whose individual means may exceed one. Those cannot simply be substituted into an individual-source anytime claim.
- Generic p-to-e conversion may be structurally powerless. The e-value memo's finite-rank caps show that some calibrators cannot reject a sparse-anomaly batch even with perfect ranking. Calculate the bound before spending compute.
- Arithmetic averaging loses peak evidence; fixed look budgets reduce early evidence further. Choosing weights on the observed source or repeatedly tuning on 53 anomalies breaks the stated simple argument.
- Four valid marginal look e-values cannot be multiplied as conditional e-factors. This is not a likelihood-ratio martingale.
- A fixed family of objects with stopped evidence differs from nightly growing survey populations. Repeated batch e-BH is not e-LOND or another proved online FDR algorithm.
- The generic ingredients are established (Vovk conformal e-prediction; Bashari et al. NeurIPS 2023; Wang–Ramdas JRSS B; Xu–Ramdas AISTATS 2024). A paper needs a further substantive result or strong real astronomy evidence.

Verdict: retain as a prespecified comparator and theoretical stress test. Report a negative power outcome if that occurs. Do not make FDR the primary endpoint merely because e-values support it; it must match a real review queue and a defensible population model.

### B: why it is the best immediate baseline but not enough for a new methods paper

Let the entire null-source trajectory be one exchangeability unit. With a scorer and horizon normalizers fitted outside calibration, rank the running test maximum against independent calibration sources' complete-trajectory maxima. The running maximum is bounded by the complete test maximum, whose conservative rank is valid. This is clean, simple, and robust to arbitrary dependence **within** a trajectory, not arbitrary population drift.

Critical objections:

- [Jing et al., arXiv:2610.34617](https://arxiv.org/abs/2610.34617), full text Sec. 3, already gives split trajectory-max versus per-step Bonferroni for early memorization detection. Its primary-source abstract and appendix explicitly state fixed scorer and exchangeable null trajectories. The page describes a NeurIPS 2026 workshop poster, not a main-conference paper. Generic novelty is ruled out.
- FLARE itself already calibrates empirical whole-life significance. The distinction must be honest held-out calibration, fixed-score provenance, and empirical utility, not “first lifetime uncertainty for FLARE.”
- A common terminal threshold can be conservative early; one noisy horizon can dominate the full maximum. Learned horizon normalizers add a policy-selection issue unless trained outside calibration/evaluation.
- Four selected looks are not the existing every-detection replay schedule. A guarantee under a fixed observation/missingness mechanism does not cover changing follow-up policy or selective extra photometry.
- A known-mixture null guarantees marginal false-alarm probability for that population. It does not guarantee 1% per rare class, robustness to arbitrary class-prior shift, or discovery precision.
- The combined SkyPortal action is lifetime alarm **OR** a separate repeated review trigger. Validating B alone does not validate the combined review queue.

Verdict: implement as the control arm and potentially useful replacement **only after** held-out replay and maintainer review. An honest null result is still a useful deliverable. Scientific novelty should not be claimed from max aggregation or the finite-rank proof.

### C: most useful research question, but it must first become one coherent method

Two ideas are currently being grouped under C and should not be silently merged:

1. **C1: support-adaptive probability adapter.** The personal-transfer audit proposes a frozen local reference expert and `lambda=E/(E+alpha)` gate fitted by proper log loss on a separate pool, followed by untouched split-conformal calibration. This is precise and testable, but nearest-neighbor ensembling and conformal adaptation are established. Its mechanistic claim must survive constant-mixture, temperature, expert-only and permuted-support controls.
2. **C2: source-level wait/classify/review policy.** This optimizes action timing and follow-up resources rather than set size. It is a broader and more valuable objective, but the policy family, risk estimand and certification procedure need to be fixed before experiments. It need not depend on C1; separating them makes attribution possible.

An implementable first C2 specification is a finite family of prefix-only stopping policies fixed on proper-training/selection data, followed by an independent calibration sample. For each fixed policy, count accepted sources and wrong accepted labels. Under i.i.d. source trajectories, a one-sided binomial bound conditional on accepted count bounds the selected error probability; simultaneous correction across the finite policy family permits choosing a policy after computing its bounds. This is a Learn-then-Test/risk-control application, not a new theorem. All the feature/scoring/model choices that define the family must be independent of that calibration sample. If too few sources are accepted, the bound is uninformative and the policy cannot be certified.

For simultaneous prediction sets, an alternative path-level class nonconformity can protect stopping at a singleton against unconditional wrong-stop error, but that is **not** an upper bound on error conditional on selection. Do not confuse these two C2 targets. Class-specific certification may be impossible for TDE/SLSN with the existing counts.

Verdict: phase-two scientific development, contingent on an honest pilot and availability of representative follow-up labels. Reward comes from demonstrating an information-matched utility improvement and exposing how measurement quality affects trustworthy decisions—not from attaching a new acronym to a standard guarantee.

The conditional literature audit strengthens the prior-art caveat: **Learn then Test** is verified in *Annals of Applied Statistics* (2025, DOI 10.1214/24-aoas1998) and explicitly covers selective/nonmonotone risks. **Active, anytime-valid risk controlling prediction sets**, NeurIPS 2024 ([arXiv 2406.10490](https://arxiv.org/abs/2406.10490)), already combines e-processes, adaptive label queries, budgets and prediction-based variance reduction. “Anytime + selective + active follow-up budget” alone is therefore not sufficient novelty. Its predictable querying and propensity-support assumptions matter for astronomy.

## Review of the actually frozen pilot, before its results

The recorded protocol is a defensible **smaller** study than the three long-term research candidates. It uses label-independent common five-fold source assignment; for each run, three folds train, one calibrates, one evaluates; the same IDs are used at all four horizons. Calibration/evaluation scores use the same retained model. Energy normalization is fitted on proper-training early-stop rows. The old test is excluded from scoring. These choices directly address the major provenance and fold/refit defects.

Its A comparator is one split per outer run, not the full five-model derandomized ensemble originally suggested by Claude. It tests fixed-look rank-e spending and e-BH, not every original A ingredient. Its C comparator is a **fixed hand-specified count gate**, `nobs/(nobs+lambda)`, not a learned correctness model or the donor project's local-reference expert. Those are appropriate pilot simplifications; report them literally.

The finite family of 45 source-level stopping policies and the `.05/(5*45)` binomial certification allocation can support simultaneous calibration-sample confidence without requiring independence across outer runs. Optimizing selection fraction among already certified policies preserves their simultaneous risk certificate, although it does not certify the chosen policy's utility estimate. The confidence-only comparison uses the same 45-way correction, avoiding a multiplicity advantage for one family.

Remaining interpretation requirements:

- The alarm comparison fixes the **same nominal** lifetime target `.01`; it does not match the **achieved** false-alarm fractions. Different conservative procedures can have different realised FAR. Any observed recall difference therefore needs both FAR and recall reported, not “higher power at equal actual FAR.” A later budget-matched experiment would need thresholds selected on independent selection data, not retrofitted from these outcomes.
- The gate comparison tests the specific three count penalties and fixed threshold/earliest-look grid. A negative finding does not refute all evidence-adaptive policies; a positive finding does not establish the local-reference support mechanism of TGLRec.
- A known-mixture alarm and pooled selected-error certificate do not establish a rare-class guarantee. Reporting rare-class rates is necessary, not certification.
- Five reuses of the same 53 anomalies must stay descriptive repeated-model sensitivity checks; there are still only 53 distinct sources.
- These are statistical statements under explicit hypothetical population assumptions. Passing exact-rank/binomial unit tests does not make a repeatedly inspected historical dataset into a fresh randomized validation sample.

## Completeness critic: required before interpreting the new pilot

1. **Recover versus regenerate.** The recovered branch does not establish Claude's latest h5/h10 nested predictions or saved fold models. Inventory exact files and hashes; either recover them or regenerate explicitly. Never write “continued the completed caches” from the chat quote alone.
2. **One source split across all horizons.** Horizon caches may independently drop feature failures and regenerate folds. Join by object ID and use a master source assignment. All synthetic context/cadence variants of a source share its role and fold.
3. **Same deployed scorer.** A model scoring calibration sources must be the exact model scoring evaluation sources, including training-only normalizers, score combiner and early stopping. Calibrating OOF energies against a later refit is excluded.
4. **Calibrators fit on training only.** Quantile/energy normalization, feature selection, horizon weights, bin edges, gates and policy grids consume information. “Unlabelled” calibration tuning does not automatically preserve split-conformal symmetry.
5. **No hidden source exclusion.** Missing or insufficient early looks need a predeclared contribution and population definition. Do not restrict to complete, easy trajectories after inspecting performance.
6. **Choose the null and decision unit.** Specify mixture-known, class-known, classifier-error, or novel-source null. Count each source once. State whether review means a primitive alarm or the entire deployed OR rule.
7. **Separate targets.** FAR, selected error, FDR, mFDR, coverage, set size and alerts/day are not interchangeable. One experiment's bound cannot be used as another's label.
8. **Finite sample feasibility.** Compute minimum rank/evidence and accepted-count requirements. A 1% target may have insufficient calibration resolution or certifiability, especially per rare class. Report exact calibration sizes.
9. **Strong comparisons.** Same model predictions, object roles, score information and score direction; fixed-look, naive repeated, Bonferroni, trajectory max, fixed e-budget, and current lifetime method where genuinely reproducible. Comparisons must match lifetime FAR and/or resources as appropriate. A score threshold change cannot beat optimally thresholding that same score's ROC.
10. **All uncertainty units are sources.** Do not bootstrap visits or dropout views. Fixed-prediction bootstrap omits model/calibration fitting uncertainty; report this and use outer/repeated split variation where possible.
11. **Old test remains spent.** Public benchmark and 53 anomaly objects were used repeatedly. Directory sealing is useful access discipline, not renewed independence. New results are exploratory/descriptive until an untouched external or time cohort is obtained.
12. **Prospective labels are delayed and selected.** Any online ACI/refresh extension needs a label availability log and a representative calibration mechanism. Selectively acquired spectra cannot be treated as an unbiased random sample of all alerts.
13. **Reproducible completion.** Save exact config, code commit/hash, data membership/hash, model hashes, dependency versions, random seeds, all candidates and negative outcomes, exit status and completion marker. Do not hide failed fitting/feature objects.
14. **Evidence boundary.** Unit tests can establish code behavior and finite-rank identities; they do not prove empirical superiority or deployment exchangeability. A paper needs scientific results beyond passing tests.

## Go/no-go gates

- **Now:** complete A/B finite-rank and boundary tests; freeze a fixed-parameter, model-matched development comparison; generate a report including failed/negative comparisons and uncertainty. This is a legitimate maintainer-facing research artifact even if neither method increases recall.
- **Then:** pursue C only if a clear operational failure/utility opportunity survives strong matched-budget baselines. Freeze its score adapter and policy family, then use independent calibration and full outer evaluation.
- **Before a publication claim:** obtain fresh representative calibration/evaluation trajectories with actual deployment context/cadence and label availability; reproduce on at least one additional model or relevant population if making a general-method claim; verify novelty against the directly overlapping papers. New methods-level novelty and high-venue acceptance are not established by this review.
