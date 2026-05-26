---
name: daily-report
description: 日结汇报 skill. Use when an experiment, writing, coding, review, or analysis agent finishes a work session and must produce a concise standardized daily report for the project curator. The report should summarize files read, experiments run, results added or modified, evidence supporting the mainline, appendix/boundary/negative evidence, claims that still cannot be made, recommended next work, and important file paths.
---

# 日结汇报 / Daily Report

Use this skill at the end of a work session as a working agent. The goal is to produce a short, curator-ready report that can be consumed by the `update-report` skill.

## Output Location

If the project has a curator workspace, save the report under:

```text
mainline_agent/agent_reports/
```

Use this filename pattern:

```text
YYYYMMDD_agentname_daily.md
```

Examples:

```text
20260520_codex1_daily.md
20260520_codex2_daily.md
```

If the user gives a different output path, follow the user path.

## Required Template

```markdown
# Daily Report: <agent name>

Date: <YYYY-MM-DD>
Role: <experiment / writing / coding / review / analysis>
Session Goal: <one sentence>

## 1. Files Read

- `<path>`: <why it was read>

## 2. Experiments Or Commands Run

- `<command or run dir>`: <purpose/status/result>

If no experiments or commands were run, write:

`No experiments or commands run.`

## 3. Results Added Or Modified

- `<path>`: <what changed>

If nothing changed, write:

`No files/results modified.`

## 4. Evidence Supporting Mainline

- <claim or result>
  Evidence: `<path>`

## 5. Appendix / Boundary / Negative Evidence

- <result>
  Placement: <appendix / boundary / negative / internal>
  Evidence: `<path>`

## 6. Claims Still Not Allowed

- <claim that must not be written>
- <claim that remains unsupported>

## 7. Recommended Next Work

- <smallest useful next step>

## 8. Important Paths

- `<path>`
- `<path>`

## 9. Open Risks Or Questions

- <risk/question>
```

## Rules

- Keep the report concise and factual.
- Prefer concrete paths, tables, figures, commands, and run directories over narrative.
- Separate mainline evidence from appendix, boundary, negative, or internal evidence.
- Mark partial, failed, exploratory, or weak-budget results clearly.
- Do not exaggerate results or upgrade evidence strength.
- Do not invent citations, metrics, experiments, or file paths.
- Do not recommend new experiments unless they are needed for a specific unsupported claim.
- If a section has no content, write a short explicit `None` or `No ...` line.
- End with a brief status sentence: `Ready for curator update.` or `Blocked: <reason>.`

## Handoff To Curator

After creating the report, tell the user:

- the report path
- whether files/results changed
- the one most important curator action

