# Research notes

Development experiments behind the forecast and anomaly layers. None of these is an
external validation; each guide records the registered criteria and the outcome.

- **Fixed-blend forecast** ([supervised_improvements.md](supervised_improvements.md)): lowers the
  median object error by 7.7% at day 10 and 7.6% at day 20 against paired refits of the grid
  recipe (five-fold BTS development evaluation). Use `forecast_recipe="blend"` in `analyze()` or
  `--forecast-recipe blend`. Some class-level errors increase within the registered tolerance.
- **Robustness by catalogue peak year** ([robustness_followup.md](robustness_followup.md)): the
  forecast gain holds when fitted on older cohorts (6.9% / 8.2% at day 10 / 20). A three-seed
  classifier raises the pooled F1 from 0.8241 to 0.8322 but does not pass its uncertainty criterion.
- **Colour-leak correction, 2026-09-17** ([external_forecast_and_colors.md](external_forecast_and_colors.md)):
  early colours are rebuilt from the observed prefix only. Corrected out-of-fold inference keeps
  7.8% / 7.6% forecast gains; on a frozen external sample the day-10 gain holds and the day-20
  interval includes zero.
- **Self-supervised anomaly representation** ([ssl_ad.md](ssl_ad.md)) and **class-conditional
  predictive anomaly scores** ([predictive_ad.md](predictive_ad.md)): both failed their adoption
  criteria; the production anomaly layer is unchanged.
- **Unified analysis results** ([unified_results.md](unified_results.md)): early-horizon
  classification, forecast interval grouping and the combined anomaly score.
- **Conditional calibration and the review trigger** ([uq_conditional_calibration.md](uq_conditional_calibration.md),
  design in [uq_preregistration.md](uq_preregistration.md)): the class-only 90% sets cover the most atypical
  fifth of SN Ia about half of the time, AGN/TDE without context at 0.26-0.30 and sparser photometry at 0.77.
  Energy-stratified sets repair this but fail the efficiency rule and are kept as a research option. The
  conformal clause of `anomaly_review` (about 8% of known objects per analysis at day 100) now fires at 1%.
