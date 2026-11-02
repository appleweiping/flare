# Verified primary-source reading: FLARE, ASTRANet, and the deployment ecosystem

Read on 2026-11-02. This is a source audit and research interpretation, not a reproduction of the papers' empirical results. No test arrays or sealed predictions were opened for this reading.

## Sources and exact versions

| Source | Version inspected | Primary URL |
|---|---|---|
| FLARE paper, *Photometric classification and out-of-taxonomy detection of ZTF transients with FLARE* | arXiv:2610.30656v1, 2026-10-22 | https://arxiv.org/html/2610.30656v1 |
| ASTRANet paper, *Catching Disguised Transients with ASTRANet: Anomaly-Aware Spectroscopic Classification and Conformal Calibration* | arXiv:2607.08044v1, 2026-07-09 | https://arxiv.org/html/2607.08044v1 |
| Upstream FLARE | main HEAD `227cbff9b3b11d6cc54f42248be7fcde0be88a15`, verified with `git ls-remote` | https://github.com/applecider-ml/flare |
| User's conditional-calibration branch | `7bf56a647c9ab0f1a845fbe9220881a0bbf5474c` | https://github.com/appleweiping/flare/tree/uq-conditional-calibration |
| applecider-agents | `789b7955dc1ede4a7d7ec1a74d8d571e3528eba9` | https://github.com/applecider-ml/applecider-agents |
| skybot | `ff0ae4e4c912e45655b036a4d2918c44224ac9e9` | https://github.com/skyportal/skybot |

Both user-supplied papers are verified arXiv preprints. I have not verified journal acceptance and do not label them published journal articles. The user's last malformed URL was interpreted as `https://github.com/skyportal/skybot`; the appended Chinese sentence is not part of its repository name. Full HTML and extracted text are retained beside this note; reference clones are under `/tmp/flare-applecider-agents` and `/tmp/flare-skybot`.

## 1. The scientific task and what FLARE actually does

FLARE is an observational decision-support pipeline: many photometric alerts compete for limited human inspection and spectroscopic follow-up. A correct research objective needs to reflect that resource allocation, beyond improving one aggregate classification score.

The benchmark has 10,497 spectroscopically confirmed objects in six known classes (SN Ia, SN CC, SLSN, AGN, TDE, CV) and 53 objects outside that taxonomy. These are BTS-selected sources, not an unbiased sample of all survey alerts. Spectroscopic selection depends on brightness; the unclassified population is about half a magnitude fainter. AGN/CV entries are the examples that passed an extragalactic transient selection. This selection function matters for exchangeability and for claims about a deployment stream (FLARE §§II.1, X).

The inference chain (FLARE §§III.1–III.4) is:

1. Causal photometric prefixes supply 160 per-band, colour, Bazin-fit and cadence features. Missing bands and failed fits remain NaN.
2. Optional host, photometric-redshift-derived luminosity, Gaia/WISE, and pre-outburst-counterpart blocks add physical information that the light curve cannot determine alone. Each block is randomly blanked on 30% of training rows.
3. A five-class LightGBM top head separates SN Ia, SN CC+, AGN, TDE, CV; a binary branch splits SN CC+ into SN CC and SLSN.
4. Class-conditional `1-P_c` scores produce prediction sets. The package also exposes class p-values, credibility, confidence, and Venn–Abers intervals.
5. A distinct anomaly model consumes a deliberately narrower feature space. It omits photometric-redshift and counterpart blocks because those improve classification but degrade blind anomaly recovery. Fourteen tree-compatible LUNA scores combine by mean percentile rank without anomaly labels; supervised combiners can exploit later confirmations.
6. A phase-specific model (5/10/20 days or full curve) feeds the SkyPortal/Fritz analysis service, which returns evidence, triage and calibrated anomaly alarm information.

The paper's macro F1 rises from 0.72 ± 0.03 with light curves alone to 0.83 ± 0.02 with all context; most of the increase comes from host information (§IV.2, Table 2). This makes a wholesale neural-model replacement an unjustified default. A UQ project can add value while holding feature extraction and fitted classifiers fixed.

Known-class uncertainty and novelty are different targets. A singleton can confidently contain the wrong known label for an out-of-taxonomy object. Conversely an ambiguous known-class object can deserve an additional observation without being scientifically novel. An empty set is not a validated probability that a new class has been discovered.

## 2. Three precise gaps stated in the FLARE paper

### Cross-fit calibration lacks the advertised split-conformal theorem

FLARE §IV.3 explicitly says its cross-fit construction is related to cross-conformal prediction and reports empirical coverage **without claiming a finite-sample guarantee**. The full-curve classifier is fitted on pooled training and validation data, whereas its calibration scores come from fold models. These are different score-producing functions. The branch's later nested study documents the resulting scale mismatch.

At nominal 90%, the paper reports 90.2% marginal coverage, with per-class values 89.8%, 91.1%, 96.8%, 90.0%, 86.8%, 89.8%. The TDE class has only 68 objects. Calibration scarcity is real: a class-conditional finite 90% threshold needs at least nine calibration observations. Merely adding cells makes that problem worse.

### Deployment photometry shifts both false alarms and set coverage

FLARE §IX.2 compares photometry of the same sources: median detections increase from 26 in BTS to 93 on Fritz; only 39% of g/r epochs overlap. On 1,555 paired objects, macro F1 changes from 0.836 to 0.807 and the nominal 1% anomaly false-alarm rate rises from 0.6% to 2.4%. Recalibrating anomaly ranks and detection-count-dependent p-values on held-out Fritz examples largely fixes anomaly rates without changing classifications.

**That recalibration does not fix classification prediction sets.** Full-curve nominal 90% sets cover only 83% of the Fritz test objects. The paper expressly requests calibration of these sets on representative Fritz photometry. An offline thinning/dropout simulation is useful stress testing, but cannot establish that this actual deployment shift has been solved.

### Lifetime alarms are empirical, not anytime-valid

FLARE §IX.3 reports that a nominal 1% per-analysis alarm flags about 10% of known sources at least once over 100 days, roughly five false alarms per week at 6.8 arrivals/day. It calibrates a whole-history statistic against replayed known sources and translates its tail probability to a false-alarm rate per unit time.

The paper explicitly describes this as empirical calibration for the replay schedule and 100-day horizon, **not an anytime-valid p-value**. In a two-fold replay of 1,882 known sources the weekly/monthly/quarterly thresholds flag 39/14/11 objects; the weekly target matches, while the quarterly target does not. A theoretical time-uniform layer addresses a real stated gap, but it must clearly specify admissible looks, fixed weights, calibration exchangeability and the population of hypotheses.

## 3. ASTRANet: prior work to build on, not rediscover

ASTRANet uses spectra rather than FLARE's light curves. Its observer-frame hierarchical spectral classifier uses optional redshift/phase metadata, a six-member deep ensemble, and a learned nonlinear combination of 16 uncertainty/distance/density/hybrid anomaly scores. It studies 289 rare/out-of-taxonomy spectra across 11 families (ASTRANet §§III–V). Its physics-based failure modes include confidently misidentified near-neighbour spectra, genuinely ambiguous cases and far-OOD objects.

Already present in that work:

* Conformal calibration of anomaly scores, including partitions by predicted class (§VI.1).
* Adaptive class prediction sets (§VI.2).
* Ensemble entropy decomposition and per-class ensemble probability intervals (§§VI.3, VI.5). The intervals `mean ± standard deviation` describe ensemble dispersion; they are not automatically calibrated confidence intervals.
* Separate diagnostics of in-distribution classification errors and out-of-taxonomy detection (§VI.6).
* AD-orthogonal abstention: commit, abstain or alert based on separate classification-confidence and anomaly axes (§VI.7). Discarding the most uncertain 10% gives 0.912 selective accuracy versus 0.866 overall in that paper's dataset; this is not FLARE evidence.
* AD-stratified Mondrian conformal calibration (§VI.8), crossed with predicted class, five score bins and fallback for cells smaller than ten. It reports Q5 coverage moving from 0.824 to 0.9534 at nominal 0.95.

Consequently, “use anomaly scores to stratify calibration,” “combine uncertainty and anomaly detection,” “add an ensemble” and “abstain on uncertain examples” are prior ingredients, not sufficient new contributions. A new FLARE study needs a distinct target and valid construction, such as cross-fit-matched time-uniform risk accounting, and empirical advantage against the corresponding prior methods at the same operational cost.

The source itself cites Yuksekgonul, Zhang, Zou and Guestrin, *Beyond Confidence: Reliable Models Should Also Consider Atypicality*, arXiv:2305.18262, for atypicality-stratified calibration. That original work must be included in a novelty comparison; venue details should be verified separately rather than inferred from this reference.

## 4. Claims requiring correction or careful interpretation

The papers are evidence, not automatically valid mathematical guarantees.

* ASTRANet §I and §VI.4 calls `1-p_M` a calibrated per-object probability of being wrong. A conformal p-value's super-uniform null property does **not** imply posterior correctness/error calibration. Even a perfectly uniform null p-value yields `1-p_M` with mean 1/2, not the classifier's error probability. Use terms such as conformity rank/tail evidence; estimate an error probability with a separate valid target and calibration protocol if needed.
* ASTRANet calls its predicted-class Mondrian rate “FDR control” in §VI.1. A per-object type-I error guarantee does not establish batch or online false discovery rate. A multiple-testing procedure and an explicit family of hypotheses are necessary.
* ASTRANet's phrase “valid under arbitrary distribution shift provided the calibration set remains exchangeable” does not remove the exchangeability condition. An uncorrected shift between calibration and evaluation distributions generally breaks it. This must not be inherited as an unconditional robustness claim.
* OOF empirical p-value histograms close to uniform do not prove exchangeability. A rigorous validation should distinguish a fixed score trained independently of calibration/evaluation from scores pooled across different learned models.
* ASTRANet §VI.8 uses calibration-fitted bin edges, finite cells and sparse-cell fallback. Its displayed simple `Quantile_(1-alpha)` formula is not by itself a complete finite-sample algorithm. Independent partition fitting or an appropriate symmetric construction, conformal finite-sample correction, and an explicit limitation for fallback cells are needed before repeating the stated conditional guarantee.
* Class coverage is the frequency with which a set includes the true class among objects of that true class. It is not precision of the predicted label. FLARE §IV.3 correctly emphasizes this distinction.

These observations justify a careful validity audit; they are not a reason to dismiss the empirical results or imply that a complete paper review has been performed.

## 5. What the supporting repositories add

### applecider-agents: orchestration and scientific interpretation

The inspected revision supplies deterministic object/model tools, a typed `Report`, an LLM tool-use loop, a model registry, an MCP server and a posting gate. Its README explicitly says every reported number comes from a tool call. The FLARE package should own numerical UQ; agents should propagate its meaning and provenance.

Useful concrete paths at commit `789b7955...`:

* `applecider_agents/flare/tools.py:26–42`: caches phase-matched calls through `flare.skyportal.classify`. `flare_classify` returns prediction sets, p-values, Venn–Abers intervals and measured coverage.
* `applecider_agents/flare/triage.py`: interpretation guide and deterministic baseline. It distinguishes ordinary, needs_spectrum, anomaly_review, already_classified and other outcomes, and checks history/context.
* `applecider_agents/core/report.py`: typed evidence/report schema. New UQ fields would need defined semantics, method/version, calibration population and validity status rather than a new prose confidence score.
* `applecider_agents/core/loop.py:22–53`: LLM loop; “blind” evaluation removes catalogue/TNS tools so true labels are not retrieved during assessment.
* `applecider_agents/core/eval.py`: separate known/OOD object lists and comparisons. An OOD-only evaluation cannot measure false-alarm burden; a mixed, labelled stream is needed.
* `applecider_agents/core/skyportal_client.py:35–45`: dry-run versus posting. The package has its own minimal REST client; this revision does not depend on skybot.

There is an existing semantic integration defect: `flare/tools.py:65` claims the set carries a “90% per-class guarantee,” and `flare/triage.py` repeats that guarantee. This conflicts with the FLARE paper's explicit empirical-only cross-fit claim and its 83% Fritz result. The old triage code also still routes on argmax-excluded/credibility rules rather than the user's newly calibrated review threshold. Before integrating new UQ, downstream clients must consume updated numerical decisions and valid caveats, not recreate obsolete thresholds in prompts.

The README mentions an eventual ASTRANet folder; the inspected packaged implementation contains FLARE plus the shared infrastructure. Do not describe ASTRANet agent support as already implemented.

### skybot: generic transport, not an uncertainty method

At commit `ff0ae4e4...`, skybot is a small `requests`-based SkyPortal API client. `skybot/client.py` exposes sources, photometry, spectra, classifications, comments, annotations, observing runs and follow-up requests; tests default to offline checks, with live integration opt-in. It can supply a future read-only deployment dataset through authorized access, and later transport approved annotations. It contributes no conformal/e-value logic or calibrated confidence model.

## 6. Research implications

The strongest evidence-supported objective is to improve **decision reliability under repeated, partial observations**, holding FLARE's effective feature/classifier backbone fixed. Compare against the shipped per-look anomaly rates, empirical whole-life replay calibration, fixed split conformal, fold-matched conformal and simple multiplicity corrections. Report false review burden, discovery power, delay, set efficiency and rare-class behavior separately.

Independent test objects are essential for any performance claim after selecting methods on the repeatedly inspected BTS benchmark. Fresh labelled Fritz photometry would directly address the paper's stated deployment gap. Re-sealing already inspected benchmark outputs does not create a fresh test set.

A class-hypothesis e-value and a known-population anomaly e-value answer different nulls. Taking an e-BH rejection of some class label as an astronomical novelty discovery would conflate these goals. A valid implementation must define the hypothesis per source, aggregation across horizons/folds, and stopping/selection rules before counting discoveries. A useful negative result on power would still be scientifically informative; theoretical validity alone does not establish operational usefulness.
