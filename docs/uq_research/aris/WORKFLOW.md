# ARIS execution and adaptation record

Pinned upstream: `wanshuiyin/Auto-claude-code-research-in-sleep` at `2927b40b2ea37774de53763338c5e8f91a7601e6`. Actual installed Codex-native sources, provenance, complete root README/installation reading receipts and project-local installation details are recorded in [bootstrap.json](bootstrap.json) and [INSTALLATION.md](INSTALLATION.md). [SKILL_READ_RECEIPT.json](SKILL_READ_RECEIPT.json) identifies the agent that fully read each applicable stage; reading was divided across the team, not falsely attributed to one agent.

## Authorized scope and runtime

The user authorized autonomous literature/design/pilot/implementation/experiments/review, with Astra research thinking and Sol execution. The user excluded paper writing and contacting others. Accordingly paper drafting/compilation, external writing workers, notifications and HTML rendering were not activated. Protocols, code documentation, numerical research logs and review receipts are implementation evidence, not a paper draft.

Native `collaboration.spawn_agent`, `send_message`, `followup_task` and `wait_agent` substitute for ARIS abstract spawn/send-input interfaces. There is no claim that Claude, a Codex MCP bridge, an external API or provider authentication was installed. Requested model identities are recorded; the host returns task IDs but no independently verified exact backend model telemetry. All semantic reviews are same-family/provisional. The original literature/generation worker used requested Astra/high, disclosed in its trace; the final separate novelty/candidate review used requested Astra/xhigh, and fresh integrity/claim review requested Astra/ultra. Sol was not used to brainstorm: the user override takes precedence over the optional second-model generation recipe.

The existing local CPU environment and Python virtualenv were reused. These cached-score experiments require no GPU; GPU allocation, remote deployment and the skill's GPU-specific examples are inapplicable. Two runner processes each evaluate a finite grid internally; 1,800 result cells are not 1,800 independent scheduled jobs. No remote queue or paid infrastructure was started. Environment variables and exact invocations are saved in execution receipts. Available primary HTTPS/arXiv/Crossref/JMLR retrieval replaced absent search connectors, with search coverage limitations disclosed.

## What actually ran

1. **Research literature and ideas:** twelve principal papers extracted and existence-verified, eight analytic lenses generated, all retained for review. Primary literature revealed exact construction collisions. This was a continued research project, not a from-scratch idea search.
2. **Novelty and critical review:** a separate xhigh Astra reviewed all eight candidates and returned PROCEED WITH CAUTION, 5/10, scoped to an empirical pilot. It ranked L1→L3, with explicit L2 denominators. The ranking occurred after Block1 outcomes were available and is not preregistered. Publication-facing refinement remained REVISE at 7.25/10.
3. **Refinement and experiment bridge:** the problem anchor, small mechanism and frozen exploratory plan were saved. Sol implemented interfaces/tests; Astra pre-run review required stronger per-deadline and confidence-priority baselines plus common time features. Amendments occurred before the affected experiment outcomes. All 480 fixed-deadline and 1,320 allocation cells completed with saved arrays/models/action logs.
4. **Review iteration:** initial design, revised fixed-deadline code, revised scheduler/continuation code, and post-pilot interpretation were reviewed. This adapts the non-writing portions of the auto-review workflow; it is not a completed paper-quality loop and did not attempt to drive an advisory score to publication readiness.
5. **Fresh integrity and result-to-claim audit:** a separate Astra/ultra read raw code, source labels, outputs and final claims; performed independent numerical reconstructions; and wrote scoped verdicts. Same-family review does not establish external truth or cross-family acceptance.
6. **Public external feasibility:** a separately frozen deterministic cohort was fetched without model scoring. All 278 candidates failed original eligibility; the known-quality control passed. This candidate frame supplies a concrete data blocker, not an excuse to defer the completed implementation.

## State and gates

The actual pinned `run_state.py` recorded the five idea-discovery stages; `idea_discovery_gate.py` ran successfully. [ARIS_RUN_STATE.json](ARIS_RUN_STATE.json) is the portable snapshot. Novelty and research-review receipts are **provisional**, scoped to proceeding with exploratory work. The evidence-gate PASS means required stage artifacts, nonempty sections and positive proceed-pilot receipts exist. It does not verify backend telemetry, prove novelty, accept scientific claims or establish publication readiness.

The bootstrap initially labeled the generic executor Sol. Before stage receipts, this was corrected to Astra for scientific discovery phases, with Sol separately identified as implementation/execution. The correction and requested-only identity source are explicit in the state; no cross-family receipt was created.

Raw review traces remain project-local under `.aris/traces/` as instructed by ARIS privacy conventions. Portable substantive reviews and metadata receipts are in [review-stage/](review-stage/DESIGN_REVIEW.md). Source state is `.aris/runs/uq-anytime-2026-11-02.json`. Composed project reports live under `docs/uq_research/aris/` to keep this research extension separate from earlier studies. The results portable manifest indexes numerical evidence; source/reading/state receipts index workflow evidence. No source skill was edited to obtain a positive gate.

## Stop criterion

Finish the frozen experiments, correct concrete audit defects, preserve working interfaces and reproducible evidence, and report the scientific verdict without upgrading it. This execution criterion is distinct from accepting an efficacy, novelty, deployment or publication claim. Mixed/partial results remain mixed/partial. No writing-stage handoff follows.
