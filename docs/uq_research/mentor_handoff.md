# Mentor handoff and external validation plan

Prepared for discussion with Argyro Sasli and Michael Coughlin. The email below is a draft; no message has been sent.

## Draft email

**Subject:** FLARE UQ: reproducible pilot, rare-class failure, and next validation step

Dear Argyro and Professor Coughlin,

I have completed a literature and implementation audit of FLARE's uncertainty layer and a reproducible development pilot with new, properly separated training and calibration splits. The pilot evaluates the same objects at days 5, 10, 20 and 100, including the effect of making repeated decisions as photometry arrives.

The main result is a practical tradeoff. Repeated uncorrected per-look alarms flag 225 of 8,922 known development objects (2.52%); Bonferroni and calibration of the complete trajectory reduce this to 71 (0.80%) and 75 (0.84%). However, both recover very few of the previously inspected anomalies. The e-value approaches are even more conservative in this experiment. These results support correcting repeated-look errors, but do not yet establish a useful discovery improvement.

A separate classification-policy diagnostic exposes a more concerning issue: an apparently good pooled accepted-label error rate of 2.86% hides 18 errors among 19 accepted true SLSN objects, and 5 among 29 accepted true TDE objects. The observed aggregate result is therefore insufficient to justify safe rare-class decisions. I would like to prioritize an objective that makes this tradeoff explicit.

The [protocol](../uq_anytime_protocol.md), [results and follow-up status](../uq_anytime_results.md), [repository/paper review](repository_and_papers_zh.md), and [verified literature/design review](literature-conditional.md) are available in the branch. The implementation and tests are ready for local review; the historical benchmark remains development evidence, and I am not treating this as a completed publication-level result.

Could we decide whether the primary operational goal should be discovery of unfamiliar transients, reliable early classification, or protection against specific rare-class errors? The next useful validation would be a temporally separated, labeled cohort with photometry as received by the service, catalog availability times, and identifiers for checking training overlap. A documented replay cadence and review/label-selection policy would let me evaluate the corresponding operational loss honestly.

Best regards,
Weiping

## What the pilot supports

The fixed pilot uses five outer fits with four horizon models each, 8,922 held-out development known objects, and 53 already inspected anomalies. The known-object figures below pool the held-out development predictions; overlapping model-training sets mean that a naive independent-binomial confidence interval for this pooled result is not justified automatically.

| Method | Known objects flagged | Empirical object-level false-alarm rate | Anomalies flagged in each of the five fits |
|---|---:|---:|---|
| Uncorrected per-look alarms | 225 / 8,922 | 2.522% | 4, 3, 3, 4, 4 |
| Bonferroni across looks | 71 / 8,922 | 0.796% | 0, 0, 0, 1, 0 |
| Complete-trajectory calibration | 75 / 8,922 | 0.841% | 0, 0, 0, 1, 1 |
| Fixed-budget e-value alarm | 17 / 8,922 | 0.191% | Zero in every fit |
| e-BH | 0 / 8,922 | 0% | Zero in every fit |

The same 53 anomalies recur across fits. They are not 265 independent discoveries, and these counts do not measure stream FDR. A method making no discoveries has not demonstrated useful discovery power.

The confidence-only and support-gated classification policies selected the same 6,725 objects, with 192 errors (2.855%) and a mean decision time of 40.82 days. Among 132 true SLSN objects, only 19 were selected and 18 of those decisions were wrong; among 58 true TDE objects, 29 were selected and 5 were wrong. These are diagnostics conditional on the **true** class, not estimates of the precision of predicted SLSN/TDE labels. No class-conditional risk-control claim follows from the pooled policy result. The exploratory learned-ranking follow-up is reported separately in the [results document](../uq_anytime_results.md); no outcome is presumed here.

## Decisions for the research meeting

1. **Choose the operational error.** Novelty discovery needs false alarms, discovery recall and review load; reliable classification needs error among published labels, publication fraction and delay; rare-class protection additionally needs class-specific false negatives and the cost of labeling a rare event as an ordinary transient. These targets can favor different policies.
2. **Choose the decision unit and deadline.** Fix the first-publication/review rule, allowed observation times, follow-up deadline, and treatment of sources with no usable data at a scheduled look. Distinguish a 1% per-look false-alarm rate from a 1% per-object lifetime rate and from FDR across discoveries.
3. **Agree on deployment constraints.** Specify the nightly review/spectroscopy budget, acceptable latency, whether an uncertain label may be revised, and which catalog queries or additional observations the policy can actually request.

## Requested external cohort and validation procedure

The cohort should contain objects first available after a documented model/data cutoff, with final labels held separately until the protocol and candidate policies are frozen. Request:

- Stable source identifiers, first detection and ingestion times, and complete identities of objects used to train or calibrate each deployed model. Check overlap with BTS, Fritz, prior model fits and duplicates before evaluation.
- Photometry as delivered to the service: timestamps, filters, flux/magnitude and uncertainties, quality flags, nondetections where available, revisions and ingestion delays. Reconstruct only information available at each decision time.
- Catalog context and missingness, including match quality, catalog/query versions and availability times. A value retrieved later cannot be retroactively supplied at an earlier look.
- Final labels, provenance and confidence, confirmation dates, unknown/unclassified outcomes, and a documented rule for handling ambiguous taxonomy.
- Alert cadence, repeat-processing schedule, review/spectroscopy selection rules, and the inclusion/query probabilities or sampling design when available. Spectroscopically labeled objects are a selected population; their unweighted results do not automatically represent the full alert stream.

Use separate external calibration and evaluation cohorts if recalibration is permitted. Freeze the models, feature extraction, policy, comparison baselines and endpoints before accessing evaluation labels. Preserve object-level grouping across every prefix, mask and repeated alert. Compare methods at equal review budget and report rare-class losses alongside pooled accuracy, latency and abstention. Report temporal/cohort drift directly; a successful synthetic thinning or masking experiment does not establish robustness to that drift.

For planning, with zero errors among independent accepted examples, the optimistic sample floor for an upper error bound of `r`, failure probability `delta`, and `J` Bonferroni-adjusted simultaneous claims is:

```text
n >= ceil(log(delta / J) / log(1 - r)).
```

For `delta=0.05` and one claim, this is 59 accepted examples at `r=0.05`, or 299 at `r=0.01`. Six simultaneous claims at 5% require at least 94 accepted examples **per claim**, even with zero errors. These are lower bounds for the zero-error case, not a sufficient sample-size guarantee for a real study: observed errors, abstention, dependence, additional selection and class imbalance can require substantially more data. A small rare-class cell should remain explicitly uncertified rather than inherit a pooled guarantee.

The immediate deliverable is a reviewable implementation, a reproducible diagnostic study and a precise prospective validation plan. Publication claims require useful gains against matched baselines and independent validation; a negative power or rare-class result should remain part of the final report.
