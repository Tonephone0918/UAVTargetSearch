---
name: update-report
description: 更新并汇报 skill. Use when acting as a research mainline or curator agent that reads daily reports from experiment/writing agents, updates the project mainline, experiment ledger, claim-evidence map, next actions, and optionally dispatches the next assignments into worker-agent task files such as codex1.md and codex2.md. Especially useful for deciding which evidence is enough, which experiments are redundant, which results belong in the main text or appendix, what the strongest and weakest defensible claims are, and what the minimal next research action should be.
---

# 更新并汇报 / Update Report

Use this skill when the user asks for `更新并汇报`, asks you to act as a curator/mainline agent for a research project, or provides daily reports/final summaries from other agents that need to be digested into project-level updates.

## Goal

Convert scattered agent outputs into a stable research control plane:

- current research mainline
- sufficient evidence
- redundant experiments to avoid
- main-text vs appendix placement
- strongest and weakest defensible claims
- minimal next action
- optional task dispatch to worker-agent instruction files

## Default Workspace

If the project has a curator folder, use it. Preferred name:

```text
mainline_agent/
```

Maintain these files when present, or create them when the user asks:

```text
mainline_agent/PROJECT_MAINLINE.md
mainline_agent/EXPERIMENT_LEDGER.md
mainline_agent/CLAIM_EVIDENCE_MAP.md
mainline_agent/NEXT_ACTIONS.md
mainline_agent/agent_reports/
```

Common worker-agent task files, when present:

```text
codex1.md
codex2.md
```

## Inputs To Read

Read only what is needed for the update:

1. New agent daily reports or final summaries.
2. Existing curator files.
3. The specific result tables, paper drafts, figures, or audit files referenced by the reports.

Prefer source artifacts over memory when numbers or claims matter.

## Curator Workflow

1. Identify the current mainline.
   - State the core problem, method line, main comparison, and intended contribution.
   - Detect drift: new results that pull the project away from the agreed mainline.

2. Classify evidence.
   - `Sufficient for main text`
   - `Sufficient for appendix/boundary`
   - `Internal only`
   - `Insufficient / needs verification`

3. Update the experiment ledger.
   - Record file paths, run directories, metrics, aggregation caveats, and usage boundaries.
   - Mark experiments that should not be repeated.

4. Update the claim-evidence map.
   - For each claim, include status, evidence paths, allowed wording, and forbidden wording.
   - Separate strong claims from weak or partially supported claims.

5. Decide next actions.
   - Prefer the smallest action that removes the largest uncertainty.
   - Avoid new experiments if existing evidence already supports the needed claim.
   - Assign work by role if multiple agents exist.

6. Optionally dispatch tasks.
   - Only update worker task files when the user explicitly asks to dispatch, assign, or update worker tasks.
   - Use `NEXT_ACTIONS.md` as the source of truth for dispatch.
   - Update only the current assignment section of each worker file when possible.
   - Preserve fixed requirements such as daily-report switches, prohibitions, output paths, and role boundaries.
   - If no clear dispatch target exists, report the proposed assignments instead of editing worker files.

7. Report risks.
   - Overclaiming
   - Evidence/source mismatch
   - Metric aggregation mismatch
   - Repeated or unnecessary experiments
   - Mainline drift
   - Worker task drift after dispatch

## Evidence Placement Rules

Use the main text for:

- central comparisons
- results directly supporting the main contribution
- figures or tables needed to understand the core claim

Use appendix or discussion for:

- boundary results
- negative or mixed ablations
- diagnostics
- robustness checks
- theoretical support that is not the main empirical result

Keep internal only:

- pilot scans with weak budgets
- failed experiments not needed for the paper
- exploratory artifacts that would confuse the main story

## Claim Strength Labels

Use these labels:

- `Supported`: evidence is adequate for the intended wording.
- `Partially supported`: evidence supports a cautious version only.
- `Boundary support`: useful for limitations, discussion, or appendix.
- `Unsupported`: do not write this claim.
- `Needs verification`: source files or numbers need checking.

## Output Standard

When updating files, keep them concise and operational. A useful curator update usually answers:

- What is the current mainline?
- What evidence is already enough?
- What should not be repeated?
- What goes in main text?
- What goes in appendix?
- What is the strongest defensible claim?
- What is the weakest or riskiest claim?
- What is the minimal next action?
- Were worker task files updated, and what changed?

## Task Dispatch Rules

Task dispatch is the bridge from curator decisions to worker-agent instructions.

When the user asks to dispatch tasks:

1. Read `NEXT_ACTIONS.md`.
2. Identify per-agent assignments, usually `Codex1 下一步` and `Codex2 下一步`.
3. Update the corresponding worker task files, usually:
   - `codex1.md`
   - `codex2.md`
4. Keep each worker file role-specific:
   - codex1: experiment materials, evidence assets, tables, figures, appendix evidence, verification.
   - codex2: writing, English draft, related work, citations, narrative, cross-references.
5. Keep hard boundaries:
   - no new training unless explicitly requested
   - no training-code edits unless explicitly requested
   - no overclaiming
   - no upgrading appendix evidence into main results
6. Preserve each file's `daily-report` switch section.
7. If a worker task file already contains old assignments, replace or clearly supersede the old assignment rather than appending conflicting instructions.
8. After dispatch, summarize:
   - which files were updated
   - what codex1 should do next
   - what codex2 should do next
   - whether daily-report is enabled or disabled

Do not create a separate dispatch skill unless the dispatch process becomes complex enough to require its own workflow.

## Prohibitions

- Do not invent citations, metrics, or run results.
- Do not upgrade appendix evidence into a main result without explicit support.
- Do not ask for new experiments just because more experiments are possible.
- Do not erase earlier caveats unless new evidence directly resolves them.
- Do not let agent-specific enthusiasm override the project mainline.
- Do not silently overwrite worker task files without a dispatch request.
- Do not remove or flip a worker's daily-report switch unless the user asks.
