#!/usr/bin/env bash
# Quick DriveSense project health snapshot: repo identity, git state, test
# results, key artifacts, and the latest tracked phase. Safe to run anytime;
# makes no changes.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

echo "=== Repo root ==="
git rev-parse --show-toplevel 2>&1

echo
echo "=== Git status ==="
git status --short --branch 2>&1

echo
echo "=== Last 5 commits ==="
git log --oneline -5 2>&1

echo
echo "=== Remote ==="
git remote -v 2>&1

echo
echo "=== Untracked files that look suspicious (cache/logs/creds) ==="
git status --porcelain 2>/dev/null | awk '{print $2}' | grep -iE "cache|cliphist|weather|mcp|\.claude|kaggle\.json|\.env" \
  && echo "^^^ REVIEW THESE BEFORE COMMITTING" || echo "none found"

echo
echo "=== Key artifacts present? ==="
for f in \
  "docs/Individual_Project_Brief.md" \
  "docs/Rubric_Alignment.md" \
  "docs/Dataset_Research.md" \
  "docs/Responsible_AI.md" \
  "docs/Master_Plan_Status.md" \
  "README.md" \
  "requirements.txt" \
  "src/drivesense/__init__.py"; do
  if [ -f "$f" ]; then echo "  [x] $f"; else echo "  [ ] MISSING: $f"; fi
done

echo
echo "=== Tests ==="
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pytest -q 2>&1 | tail -15
else
  echo "  .venv not found — cannot run tests. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -e ."
fi

echo
echo "=== Latest phase status (docs/Master_Plan_Status.md) ==="
if [ -f docs/Master_Plan_Status.md ]; then
  grep -n "Next incomplete highest-priority task" -A 5 docs/Master_Plan_Status.md
else
  echo "  docs/Master_Plan_Status.md not found"
fi
