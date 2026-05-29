---
description: Write or update post-experiment findings and conclusions for the current feature branch into specs/<feature>/conclusions.md.
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty). Treat it as free-form notes from the researcher (headline results, things to emphasize, things to skip). Preserve the researcher's voice where quoted.

## Goal

Capture the outcome of research/experiments performed on the current feature branch in a durable `conclusions.md` alongside the other speckit artifacts. This is the counterpart to `research.md`: `research.md` is pre-work (assumptions, unknowns, planned approach), `conclusions.md` is post-work (what was actually found, what was decided, what to carry forward).

Run this after experiments have been executed — typically late in the feature, before merging the branch or spinning up the next feature.

## Operating Constraints

**SCOPE**: Only write/modify `FEATURE_DIR/conclusions.md`. Do not edit `spec.md`, `plan.md`, `tasks.md`, or `research.md` — if a conclusion contradicts them, note it in the report and let the user decide whether to revise upstream artifacts.

**EVIDENCE-FIRST**: Every claim in the conclusions file must be traceable to a concrete source — a notebook cell, a W&B run, a commit, a file under `experiments/`, or an explicit user statement. If the evidence is missing, mark the claim `[unverified]` rather than inventing support.

**APPEND-FRIENDLY**: If `conclusions.md` already exists, do not overwrite it silently. Add a new dated section and preserve prior content.

## Execution Steps

### 1. Initialize Context

Run `.specify/scripts/bash/check-prerequisites.sh --json --paths-only` once from repo root. Parse JSON for:

- `FEATURE_DIR`
- `FEATURE_SPEC`
- `IMPL_PLAN` (optional)
- `TASKS` (optional)

Derive:

- `CONCLUSIONS = FEATURE_DIR/conclusions.md`
- `RESEARCH = FEATURE_DIR/research.md`

Abort if `FEATURE_DIR` cannot be resolved — instruct user to check out a feature branch or run `/speckit.specify` first.

For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

### 2. Gather Evidence

Load lightly, in this order, stopping early once you have enough to write a credible summary:

- `spec.md` — original goal, success criteria
- `research.md` — what was planned / unknown going in
- `tasks.md` — what was actually executed (look for completed vs skipped)
- `$ARGUMENTS` — user's free-form notes, if any
- Notebooks referenced by this feature: `notebooks/*.ipynb` matching the feature keywords — scan markdown cells and final output cells; do not dump full notebooks into context
- `experiments/` directory at repo root — any per-run logs tied to this feature
- W&B run URLs or artifact names referenced in notebooks, commits, or user input (do not fetch; just cite)
- Recent git log on the current branch (`git log --oneline <main-branch>..HEAD`) for commit-level narrative

If evidence is thin, ask the user up to 3 targeted questions before writing. Do not invent results.

### 3. Decide: New File vs Append

- If `conclusions.md` does not exist → create with full scaffold (Step 4).
- If it exists → append a new `## Update — YYYY-MM-DD` section with only the sections that have new content. Preserve earlier entries verbatim.

### 4. Structure (for a new file)

Use this scaffold. Drop sections that do not apply rather than filling them with filler.

```markdown
# Conclusions — <feature slug>

**Status:** <draft | final> · **Date:** YYYY-MM-DD · **Branch:** <branch-name>

## Summary
<3–5 sentences. What was the question, what did we find, what is the recommended next step. A stakeholder should be able to read only this section and know whether to read further.>

## Experiments run
<Bulleted list. One line per experiment: name, variant, dataset/scope, where the artifacts live (notebook path, W&B run, commit SHA). No prose walls.>

## Results
<The actual findings. Tables preferred over prose where there are numbers. Cite the source for each row. If a result contradicts something in research.md or spec.md, flag it inline.>

## Decisions
<What we are committing to as a result. Each decision: "Decision: … · Rationale: … · Source: …".>

## Open questions
<Things the experiments did not settle. Each: "Question · Why it matters · Proposed follow-up (next feature, not this one).">

## Handoff
<What the next feature / next researcher needs to know. Pointers to the best artifacts (one or two, not everything). Any config, dataset, or checkpoint that must be preserved.>

## Evidence index
<Flat list of sources referenced above: notebook paths, W&B runs, commit SHAs, files under experiments/. This is the "bibliography" — makes the document auditable.>
```

### 5. Write

Write to `CONCLUSIONS` (create or append per Step 3). After writing, report to the user:

- Path written
- Whether it was a new file or an append
- Any `[unverified]` claims that still need evidence
- Any contradictions found between conclusions and upstream artifacts (`spec.md`, `research.md`)

### 6. Next Actions

Suggest, but do not execute:

- If contradictions with `spec.md` / `plan.md` exist → recommend `/speckit.clarify` or a manual spec revision
- If the branch is ready to merge → remind the user to commit `conclusions.md` before merging
- If follow-up experiments are open → suggest opening a new feature via `/speckit.specify`

## Operating Principles

- **Short over complete.** A tight 1-page `conclusions.md` that is read beats a 10-page one that is not.
- **Cite everything.** A finding without a source is noise. Prefer linking a W&B run or a notebook cell over paraphrasing it.
- **Do not rewrite history.** Appends on re-run; never silently overwrite prior conclusions.
- **No marketing tone.** This is an internal research log, not a blog post. State results plainly, including negative results.
- **Stay in scope.** Only `conclusions.md` is written. Upstream artifact edits are surfaced as recommendations, not performed.

## Context

$ARGUMENTS
