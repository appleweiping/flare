#!/usr/bin/env python3
"""Install the pinned, nonwriting ARIS subset without changing global settings.

Requires Python 3.10+, Git, Bash, and network access only when the source snapshot
is missing. Existing source and user files are verified or preserved, never
replaced. Research run state is not initialized or advanced by this installer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


COMMIT = "2927b40b2ea37774de53763338c5e8f91a7601e6"
REPOSITORY = "https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git"
ARCHIVE_SHA256 = "e2fe3ca4bae30912fcc99ab34a5a8c73558f325954ad9015038aca63d60c7b03"
TREE_SHA256 = "b257cfdacac14e868430f275dd3956915aa9044c867dfb115a12e7b966e8d5c7"
SOURCE_FILES = 767
REQUESTED = (
    "idea-discovery,research-lit,idea-creator,novelty-check,research-review,"
    "research-refine-pipeline,research-refine,experiment-plan,experiment-bridge,"
    "run-experiment,auto-review-loop,experiment-audit,result-to-claim"
)
SKILLS = set(REQUESTED.split(",")) | {
    "experiment-queue", "monitor-experiment", "training-check", "ablation-planner"
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_inventory(source: Path) -> dict[str, str]:
    """Check names, executable bits, and bytes; ignore generated Python caches."""
    digest = hashlib.sha256()
    inventory = {}
    for path in sorted(source.rglob("*"), key=lambda p: p.relative_to(source).as_posix()):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"source contains an unexpected symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"source contains an unexpected file type: {path}")
        name = relative.as_posix()
        inventory[name] = sha256(path.read_bytes())
        kind = "executable" if path.stat().st_mode & 0o111 else "file"
        digest.update(
            name.encode() + b"\0" + kind.encode() + b"\0"
            + inventory[name].encode() + b"\n"
        )
    if len(inventory) != SOURCE_FILES or digest.hexdigest() != TREE_SHA256:
        raise ValueError(
            f"source differs from pinned ARIS: {source}; found {len(inventory)} files, "
            f"tree SHA-256 {digest.hexdigest()}. Preserve it and select another --source."
        )
    return inventory


def fetch_source(destination: Path, checkout: Path, environment: dict[str, str]) -> None:
    """Fetch only the fixed commit using Git's existing HTTPS configuration."""
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True, env=environment)
    subprocess.run(
        ["git", "-C", str(checkout), "fetch", "--quiet", "--depth=1", REPOSITORY, COMMIT],
        check=True, env=environment,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "FETCH_HEAD"],
        text=True, env=environment,
    ).strip()
    if actual != COMMIT:
        raise ValueError(f"Git fetched {actual}, expected {COMMIT}")
    archive = subprocess.check_output(
        ["git", "-C", str(checkout), "archive", COMMIT], env=environment,
    )
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("pinned Git archive checksum mismatch; refusing installation")
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive path: {member.name}")
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = bundle.extractfile(member)
                if payload is None:
                    raise ValueError(f"missing archive payload: {member.name}")
                with target.open("xb") as stream:
                    stream.write(payload.read())
                target.chmod(member.mode & 0o777)
            else:
                raise ValueError(f"unexpected archive member type: {member.name}")
    source_inventory(destination)


def local_installer(source: Path, project: Path) -> str:
    text = (source / "tools/install_aris_codex.sh").read_text()
    original = 'GLOBAL_POINTER="$HOME/.aris/repo"'
    if text.count(original) != 1:
        raise ValueError("upstream installer no longer has exactly one expected pointer assignment")
    pointer = str(project / ".aris/repo")
    for character in ("\\", '"', "$", "`"):
        pointer = pointer.replace(character, "\\" + character)
    return text.replace(original, f'GLOBAL_POINTER="{pointer}"')


def validate_existing(project: Path, source: Path, installer: str) -> None:
    for relative in (".aris", ".agents", ".agents/skills"):
        if (project / relative).is_symlink():
            raise ValueError(f"refusing a symlinked installation parent: {project / relative}")
    wrapper = project / ".aris/install_aris_codex.local.sh"
    if wrapper.is_symlink() or (wrapper.exists() and wrapper.read_text() != installer):
        raise ValueError(f"preserving a different existing installer: {wrapper}")
    pointer = project / ".aris/repo"
    if pointer.is_symlink() or (pointer.exists() and pointer.read_text().strip() != str(source)):
        raise ValueError(f"preserving a different existing helper pointer: {pointer}")
    tools = project / ".aris/tools"
    if tools.is_symlink():
        if tools.resolve() != source / "tools":
            raise ValueError(f"preserving a different existing helper link: {tools}")
    elif tools.exists():
        raise ValueError(f"preserving an existing real helper path: {tools}")
    agents = project / "AGENTS.md"
    if agents.is_symlink():
        raise ValueError(f"preserving symlinked project instructions: {agents}")
    manifest = project / ".aris/installed-skills-codex.txt"
    if manifest.is_symlink():
        raise ValueError(f"preserving symlinked manifest: {manifest}")
    if manifest.exists():
        rows = [line.split("\t") for line in manifest.read_text().splitlines()]
        roots = [row[1] for row in rows if len(row) == 2 and row[0] == "repo_root"]
        if roots != [str(source)]:
            raise ValueError("existing manifest uses another source; preserving that installation")
        extra = {row[1] for row in rows if len(row) == 5 and row[0] == "skill"} - SKILLS
        if extra:
            raise ValueError(f"existing managed skills extend beyond this nonwriting subset: {sorted(extra)}")


def install(project: Path, source: Path, dry_run: bool, environment: dict[str, str]) -> None:
    inventory = source_inventory(source)
    installer = local_installer(source, project)
    validate_existing(project, source, installer)
    command = [
        "bash", "-s", "--", str(project), "--aris-repo", str(source),
        "--skills", REQUESTED, "--exclude", "render-html", "--no-doc", "--skip-new",
    ]
    # Feed the reviewed derived installer on stdin: no project writes for preview.
    subprocess.run(command + ["--dry-run"], input=installer, text=True, check=True, env=environment)
    if dry_run:
        print("Verified source and previewed installation; project files were not changed.")
        return
    state_dir = project / ".aris"
    state_dir.mkdir(exist_ok=True)
    wrapper = state_dir / "install_aris_codex.local.sh"
    if not wrapper.exists():
        with wrapper.open("x") as stream:
            stream.write(installer)
    subprocess.run(
        ["bash", str(wrapper), str(project), "--aris-repo", str(source),
         "--skills", REQUESTED, "--exclude", "render-html", "--no-doc", "--skip-new", "--quiet"],
        check=True, env=environment,
    )
    helper_link = state_dir / "tools"
    if not helper_link.is_symlink():
        helper_link.symlink_to(source / "tools", target_is_directory=True)
    inventory_path = state_dir / "source-sha256.json"
    if not inventory_path.exists():
        with inventory_path.open("x") as stream:
            json.dump({"commit": COMMIT, "sha256": inventory}, stream, indent=2)
            stream.write("\n")
    agents = project / "AGENTS.md"
    if not agents.exists():
        with agents.open("x") as stream:
            stream.write(
                "# Pinned ARIS scientific workflow\n\n"
                f"Use project-local `.agents/skills/` at ARIS commit `{COMMIT}`. "
                "Resolve helpers through `.aris/tools/`. Keep source immutable and "
                "runtime `.aris/` and `.agents/` paths out of commits.\n\n"
                "This run permits scientific research, experiments, and review. "
                "Paper-writing and publication stages are excluded; `RENDER_HTML=false`. "
                "Execution uses `gpt-6.1-sol`; scientific design and review use "
                "`gpt-6-astra`. Both belong to the same family: semantic review is "
                "`same-family` and `provisional`, never cross-family acceptance. "
                "Adapt native agent coordination to the host's available collaboration "
                "tools and record actual identities. No external providers or Goal "
                "timers are activated by setup.\n"
            )
    installed = set()
    for line in (state_dir / "installed-skills-codex.txt").read_text().splitlines():
        row = line.split("\t")
        if len(row) == 5 and row[0] in {"skill", "support"}:
            target = project / row[3]
            if not target.is_symlink() or target.resolve() != source / row[2]:
                raise ValueError(f"installed link verification failed: {target}")
            if row[0] == "skill":
                installed.add(row[1])
    if installed != SKILLS:
        raise ValueError(f"installed skill set differs: {sorted(installed ^ SKILLS)}")
    print(f"Verified {SOURCE_FILES} pinned source files and {len(installed)} scientific skills; "
          "shared references and helper link are available. Research state was not changed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", type=Path, default=Path("/workspace/.research/aris-2927b40"))
    parser.add_argument("--dry-run", action="store_true", help="verify and preview; do not write project files")
    options = parser.parse_args()
    project = options.project.resolve()
    source = options.source.absolute()
    if not project.is_dir():
        parser.error(f"project directory does not exist: {project}")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        if source.is_symlink():
            raise ValueError(f"source path is a symlink; preserving it: {source}")
        if source.exists():
            install(project, source.resolve(), options.dry_run, environment)
        else:
            if not options.dry_run:
                source.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="aris-pinned-", dir=None if options.dry_run else source.parent,
            ) as temporary:
                temporary_path = Path(temporary)
                fetched = temporary_path / "source"
                fetch_source(fetched, temporary_path / "git", environment)
                if options.dry_run:
                    print(f"Source absent: preview uses a verified temporary snapshot. "
                          f"Actual installation would retain it at {source}.")
                    # The fixed subset can be previewed against the temporary source;
                    # there is no preexisting manifest in a fresh project.
                    install(project, fetched, True, environment)
                else:
                    if source.exists() or source.is_symlink():
                        raise ValueError(f"source appeared concurrently; preserving it: {source}")
                    fetched.rename(source)
                    install(project, source.resolve(), False, environment)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ARIS setup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
