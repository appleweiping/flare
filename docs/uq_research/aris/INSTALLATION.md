# Pinned ARIS installation

The project uses ARIS commit `2927b40b2ea37774de53763338c5e8f91a7601e6` from <https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep>. The source snapshot is `/workspace/.research/aris-2927b40`, outside the Git checkout. All 767 source files were checked byte for byte against the clean pinned checkout. The snapshot includes upstream documentation and assets; only the 17 scientific skills listed in `bootstrap.json` are activated through project skill links.

The full root README, Codex README, setup guide, agent guide, 1,033-line Codex installer, and skill catalog were read before installation. `bootstrap.json` records complete-read chunk ranges and per-file SHA-256 values. `.aris/source-sha256.json` inventories all snapshot source files. Hashing alone is not recorded as reading a file.

## Reproduce from a fresh checkout

The tracked setup script reconstructs the source and local skill installation; no existing `.aris/` files are required:

```bash
cd /workspace/flare
python3 scripts/uq_anytime/setup_aris.py --dry-run
python3 scripts/uq_anytime/setup_aris.py
```

Requirements are Python 3.10+, Git, and Bash. If the default source directory `/workspace/.research/aris-2927b40` is absent, the script uses Git's existing HTTPS configuration to fetch the exact commit from the fixed upstream URL, verifies the commit ID and embedded archive SHA-256, and retains the verified source snapshot. No token entry, provider registration, or global configuration write is needed. The dry run uses a temporary verified copy when the retained source is missing and leaves the project and retained source location unchanged.

If the source already exists, the script checks all 767 source files against an embedded canonical tree SHA-256 covering relative paths, executable bits, and file bytes. Generated `__pycache__` directories are ignored. An altered snapshot is preserved and causes setup to fail; it is never replaced. Existing project files, conflicting links, different local installer copies, and installations outside the selected scientific subset are likewise preserved. A matching installation can be rerun safely. Existing `AGENTS.md` content is left byte for byte unchanged; a minimal scope document is created only when it is absent.

To use another checkout or writable source location, pass explicit paths:

```bash
python3 scripts/uq_anytime/setup_aris.py \
  --project /path/to/flare --source /path/to/retained/aris-2927b40
```

The script previews the upstream installation before applying it, creates the exact project-local pointer adaptation described below, verifies the installed 17-skill set and helper links, and leaves research run state untouched. Keep generated `.aris/` and `.agents/` entries out of commits. It does not publish, update dependencies, or launch paper stages. The current environment's Python virtual environment and research run remain separate from skill installation.

## Local installation and adaptation

The upstream Codex installer supports project-local skill symlinks and manifest tracking. It also writes `$HOME/.aris/repo` unconditionally and does not create a project helper link. To keep all writes within this project, the exact upstream installer was preserved and a derived copy at `.aris/install_aris_codex.local.sh` changes only this assignment:

```diff
-GLOBAL_POINTER="$HOME/.aris/repo"
+GLOBAL_POINTER="/workspace/flare/.aris/repo"
```

The tested install command, preceded by a dry run, was:

```bash
cd /workspace/flare
bash .aris/install_aris_codex.local.sh /workspace/flare \
  --aris-repo /workspace/.research/aris-2927b40 \
  --skills idea-discovery,research-lit,idea-creator,novelty-check,research-review,research-refine-pipeline,research-refine,experiment-plan,experiment-bridge,run-experiment,auto-review-loop,experiment-audit,result-to-claim \
  --exclude render-html --no-doc --skip-new --quiet
```

The catalog adds four scientific dependencies: `experiment-queue`, `monitor-experiment`, `training-check`, and `ablation-planner`. The resulting manifest has 17 skills and the `shared-references` support directory. All 18 symlink targets were validated. A separate `.aris/tools` symlink points to the pinned source's `tools/` directory.

`render-html` is intentionally excluded because `RENDER_HTML=false` disables its invocations in idea discovery and the review loop. The installer's generic catalog warnings about that dependency were recorded; the scientific stages remain available. Paper drafting, LaTeX, presentation, external writing-worker, and publication stages were not installed. No external provider or credentials were configured. The custom `AGENTS.md` records these scope and runtime decisions; `--no-doc` prevented the upstream generic managed block from replacing it.

The live host exposes `collaboration.spawn_agent`, `collaboration.send_message`, `collaboration.followup_task`, and `collaboration.wait_agent`. They implement the skill pack's agent coordination pattern; the host does not expose the mirror's literal `send_input` tool. Execution uses the requested `gpt-6.1-sol`; scientific design and review use `gpt-6-astra`. These are different models within the same family, so semantic review records `review_independence: same-family` and `acceptance_status: provisional`. Installing the skill pack is not cross-family scientific acceptance.

This is an ordinary chat without a host-bound Codex Goal. Goal Pulse was not installed or started. No notification, email, commit, or push was performed by the installation agent.

## Resume and readiness checks

The retained source, local links, and local configuration survive as filesystem state in the environment snapshot; no service process needs restarting. Do not update the pinned source during this research run. Future tasks use the existing isolated checkout; create a worktree only when the user requests one.

```bash
cd /workspace/flare
export PYTHONPYCACHEPREFIX=/workspace/.cache/flare/pycache
export MPLCONFIGDIR=/workspace/.cache/flare/matplotlib
export XDG_CACHE_HOME=/workspace/.cache/flare

# Repeatable install preview: initially reported REUSE=18, CREATE=0, CONFLICT=0.
bash .aris/install_aris_codex.local.sh /workspace/flare \
  --aris-repo /workspace/.research/aris-2927b40 \
  --reconcile --dry-run --exclude render-html --no-doc --skip-new

# Read real research state without advancing any stage.
/workspace/.venvs/flare/bin/python .aris/tools/run_state.py status \
  . uq-anytime-2026-11-02
```

The real `uq-anytime-2026-11-02` run was created using the official five phase names (`research-lit`, `idea-creator`, `novelty-check`, `research-review`, `research-refine-pipeline`), `--executor gpt-6.1-sol`, and `--provisional-advances`. All phases were pending when handed to the scientific leader. That leader owns subsequent stage updates. Custom artifact locations under `docs/uq_research/aris/` must be passed explicitly to the appropriate helpers. Bootstrap completion does not mark any scientific stage complete.

Validation exercised the upstream run-state, provenance, and idea-discovery evidence-gate suites:

```bash
cd /workspace/.research/aris-2927b40
PYTHONPYCACHEPREFIX=/workspace/.cache/flare/pycache \
  /workspace/.venvs/flare/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_run_state.py tests/test_provenance.py tests/test_idea_discovery_gate.py
```

Result: **58 passed**. Installer syntax, dry-run conflict detection, actual symlink installation, reconciliation, and real run initialization also passed. This establishes that the installed workflow helpers operate; scientific judgments still require their actual evidence and review receipts.
