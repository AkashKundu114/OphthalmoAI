# OphthalmoAI Git History Remediation Plan

**Document:** `REMEDIATION/HISTORY_PLAN.md`  
**Status:** **PENDING USER APPROVAL (Approval Gate 1)**  
**Target:** Removal of legacy binary database `ophthalmoai.db` and large legacy artifacts from historical commits.

---

## 1. Problem Statement

During Phase A inspection of git history (`git log --all --full-history -- "**.db"`), the binary database `ophthalmoai.db` (approx. 700 KB - 1.2 MB across revisions) was found committed in early revisions prior to commit `c33eff8` (where it was untracked and added to `.gitignore`).

### Read-Only Inspection Findings
Read-only SQLite inspection (`scripts/inspect_db_readonly.py`) confirmed:
- Zero real patient data, medical records, or genuine personal information.
- All 503 user accounts in the historical database are synthetic test fixtures with generated emails (`cli***@example.com`) and test password hashes produced during local integration test runs.
- However, committing binary SQLite databases to git history:
  1. Bloats repository clone size.
  2. Creates the appearance of unhygienic data handling to outside recruiters and academic reviewers.
  3. Violates best practices for open-source reproducible research.

---

## 2. Remediation Options & Trade-Off Analysis

### Option A: History Rewrite via `git-filter-repo` (Recommended for Clean Release)
- **Command:**
  ```bash
  git filter-repo --path ophthalmoai.db --invert-paths --force
  ```
- **Advantages:**
  - Completely purges all revisions of `ophthalmoai.db` from all commit trees and packfiles.
  - Reduces total repository clone size.
  - Preserves full commit messages, author attribution, and development chronology for all other files.
- **Risks & Impact:**
  - Rewrites all commit hashes downstream of the earliest `ophthalmoai.db` commit.
  - Existing forks or local clones will diverge and require re-cloning (`git clone`) or resetting.
  - **Requires explicit user authorization under Approval Gate 1.**

---

### Option B: Fresh History Release Branch (`orphan` branch)
- **Command:**
  ```bash
  git checkout --orphan release/public-v0.1.0
  git commit -m "feat(core): initial public release v0.1.0"
  ```
- **Advantages:**
  - Guarantees zero legacy artifacts, zero old blobs, and absolute minimal clone size.
  - Pristine audit trail starting from clean code.
- **Risks & Impact:**
  - Loses historical commit timestamps and development history.
  - GitHub contribution graphs before this date will not show historical commits on the default branch.

---

### Option C: Retain History, Exclude Moving Forward (Current Default)
- **Status:** Already implemented in commit `c33eff8` and on branch `release/public-cleanup`.
- `ophthalmoai.db` is in `.gitignore:77:ophthalmoai.db*` and has zero presence in current branch trees.
- `tests/fixtures/` provides scripted JSON fixtures for tests.
- **No force-push required; zero risk to stars/forks.**

---

## 3. Recommendation & Next Steps

1. **Immediate Action:** Proceed with Phase B, C, D, E, F on branch `release/public-cleanup` without rewriting history.
2. **Approval Gate 1 Decision:** At the final release checkpoint, the user can choose whether to execute **Option A** (`git filter-repo`) before publishing to GitHub, or publish branch `release/public-cleanup` under **Option C**.
