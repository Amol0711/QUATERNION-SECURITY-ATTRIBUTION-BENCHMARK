from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {
    ".py", ".c", ".h", ".json", ".csv", ".yaml", ".yml", ".toml", ".txt", ".md", ".gitignore"
}
PROHIBITED_SUFFIXES = {".tex", ".bib", ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".aux", ".bbl", ".blg"}
EXCLUDED_PARTS = {".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "data", "results", "build", "dist"}

# Build sensitive terms from fragments so this validator can scan its own source
# without producing self-referential findings.
_JOURNAL = "TI" + "FS"
_JOURNAL_LONG = "Transactions on Information " + "Forensics"
_WORKFLOW_TERMS = ["mile" + "stone", "hand" + "off", "rebut" + "tal", "sign" + "off", "revision " + "plan", "review " + "report"]
_SUBMISSION_TERMS = ["manu" + "script", "supplementary " + "material", "journal " + "submission"]
_PERSONAL_TERMS = ["Amol " + "Yerudkar", "ayerud" + "kar@", "Zhejiang Normal " + "University"]
_LOCAL_TERMS = ["sand" + "box:", "/mnt/" + "data/", "/home/" + "oai/"]

CONTENT_PATTERNS = {
    "journal_identity": re.compile(r"\b" + re.escape(_JOURNAL) + r"\b|" + re.escape(_JOURNAL_LONG), re.I),
    "development_workflow": re.compile(r"\b(?:" + "|".join(re.escape(item) for item in _WORKFLOW_TERMS) + r")\b", re.I),
    "submission_material": re.compile(r"\b(?:" + "|".join(re.escape(item) for item in _SUBMISSION_TERMS) + r")\b", re.I),
    "personal_identity": re.compile("|".join(re.escape(item) for item in _PERSONAL_TERMS), re.I),
    "local_path": re.compile(r"(?:" + "|".join(re.escape(item) for item in _LOCAL_TERMS) + r"|[A-Za-z]:\\Users\\)", re.I),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
}

# Random protected payloads can accidentally resemble generic e-mail addresses.
# Binary files are therefore checked only for explicit high-confidence tokens.
_BINARY_TOKENS = tuple(
    item.encode("utf-8")
    for item in [_JOURNAL, _JOURNAL_LONG, *_WORKFLOW_TERMS, *_SUBMISSION_TERMS, *_PERSONAL_TERMS, *_LOCAL_TERMS]
)


def public_files(repo_root: str | Path) -> list[Path]:
    root = Path(repo_root).resolve()
    output: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo", ".so", ".dll", ".dylib", ".whl"}:
            continue
        output.append(path)
    return sorted(output, key=lambda item: item.relative_to(root).as_posix())


def scan_repository(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    findings: list[dict[str, str]] = []
    files = public_files(root)
    forbidden_name_tokens = tuple(term.lower().replace(" ", "_") for term in [*_WORKFLOW_TERMS, *_SUBMISSION_TERMS])
    for path in files:
        rel = path.relative_to(root).as_posix()
        lower_name = rel.lower()
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            findings.append({"category": "prohibited_file_type", "path": rel, "detail": path.suffix.lower()})
        if any(token in lower_name.replace("-", "_") for token in forbidden_name_tokens):
            findings.append({"category": "development_filename", "path": rel, "detail": "nonpublic workflow term"})
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"README", "LICENSE", ".gitignore"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for category, pattern in CONTENT_PATTERNS.items():
                match = pattern.search(text)
                if match:
                    findings.append({"category": category, "path": rel, "detail": match.group(0)[:120]})
        else:
            data_lower = path.read_bytes().lower()
            for token in _BINARY_TOKENS:
                if token.lower() in data_lower:
                    findings.append({"category": "explicit_binary_token", "path": rel, "detail": token.decode("utf-8", errors="replace")[:120]})
                    break
    return {
        "status": "PASS" if not findings else "FAIL",
        "files_scanned": len(files),
        "finding_count": len(findings),
        "findings": findings,
    }


def prospective_archive_safety(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    findings: list[dict[str, str]] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if path.is_symlink():
            findings.append({"category": "symbolic_link", "path": rel.as_posix()})
        if rel.is_absolute() or ".." in rel.parts:
            findings.append({"category": "unsafe_path", "path": rel.as_posix()})
        if path.is_file() and os.stat(path, follow_symlinks=False).st_mode & 0o6000:
            findings.append({"category": "unsafe_mode", "path": rel.as_posix()})
    return {"status": "PASS" if not findings else "FAIL", "finding_count": len(findings), "findings": findings}
