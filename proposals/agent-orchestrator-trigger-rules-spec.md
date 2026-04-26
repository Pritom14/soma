# Trigger Rules — Design Spec (v1) for Agent Orchestrator

**Date:** 2026-04-17
**Status:** Draft (pending review)
**Tracking issue:** ComposioHQ/agent-orchestrator (to be filed)

---

## What this document is

This spec locks in the design decisions for adding event-driven session spawning to Agent Orchestrator (AO). It covers: how trigger rules work, how the investigator session role fits into the existing session model, how the flaky guard and failure classification prevent noise, how cycle prevention works, and what is in v1.

A companion RFC will cover the plugin interface, domain objects, config shape, persistence model, CLI commands, and API routes. This spec does not duplicate those — it references and extends them.

---

## The gap this closes

The existing inbound webhook pipeline (`/api/webhooks/[...slug]/route.ts`) already does the hard work: signature verification via `scm.verifyWebhook()`, event parsing via `scm.parseWebhook()`, project matching, and lifecycle checks on affected sessions.

The gap is one branch of code:

```typescript
const affectedSessions = findAffectedSessions(sessions, candidate.projectId, event);
if (affectedSessions.length === 0) {
  continue;  // ← event is silently dropped here
}
```

When an event arrives and **no session exists yet** — a fresh CI failure on `main`, a newly labeled issue — the event is dropped. There is no mechanism to spawn a session in response. AO currently requires a human to run `ao spawn` for every session.

There is also a second entry point: `SessionStatus` already includes a `ci_failed` value. Sessions that transition into `ci_failed` represent a persistent, confirmed code-level failure — a clean trigger point that bypasses the need for the flaky guard entirely, since the lifecycle has already confirmed the failure is real.

Trigger rules answer the question: *what should happen when a relevant event arrives and no session exists yet?*

---

## How it works: the two key questions

### What does the spawned session do?

The spawned session is an **investigator**, not a fixer. Its first pass is read-only: read the CI log, identify the failing step, classify the failure, and file a structured GitHub Issue with findings.

**Why not fix immediately?**

1. **Signal quality.** Not every CI failure is a code bug. Infrastructure outages, flaky tests, and rate limit errors look identical to code regressions at the webhook level. Spawning a fixer for an infrastructure failure wastes compute and creates noise on the repo.
2. **Trust ramp.** An investigator filing an issue is low-risk — the team can close it if wrong. An agent pushing a fix PR on an untriaged failure is high-risk.
3. **Audit trail.** An issue exists regardless of whether a fix is attempted. The team can see what AO investigated, how it classified the failure, and which files it implicated — even if no fix was produced.

Auto-fix via PR is architecturally supported but **off by default** via `on_confident_fix: issue_only`. Projects that trust the investigator can set `on_confident_fix: pr`.

### How does the trigger know it is the right event?

Each trigger rule specifies:
- The event type to match (`workflow_run.completed`, `issues.labeled`, etc.)
- Filters on that event's payload (conclusion, branch, label, step category)
- A triage config (flaky guard, minimum consecutive failures)

The rule engine evaluates rules in order and spawns for the first match. Rules that don't match are skipped. Events that match no rules are silently dropped — same as today.

---

## Decision: extend the existing webhook path, not add a new route

Two approaches were considered:

| Approach | Description | Verdict |
|---|---|---|
| **A: New route** | Add `/api/webhook/:provider` alongside the existing `/api/webhooks/[...slug]` | Duplicates verification and parsing logic. Two inbound webhook paths to maintain. |
| **B: Extend existing** | Add `findMatchingTriggers()` to `scm-webhooks.ts`, call it after `findAffectedSessions()` returns empty | Single inbound path. Additive — only fires when no existing session is affected. |

**Decision: Approach B.**

The existing route handles all the hard parts. The trigger engine is a new branch in the existing flow, not a parallel flow. Trigger rules cannot interfere with live session routing.

---

## Decision: investigator extends the existing role mechanism

AO already stores and reconstructs session roles via `readMetadata` / `writeMetadata` and `resolveSessionRole()` in `session-manager.ts`. Role is a first-class concept in the metadata layer.

The proposal is to extend the existing role set to include `"investigator"` as a new value. No new field is needed on `Session` or `SessionSpawnConfig` — the existing metadata mechanism handles it. `resolveSessionRole()` would be updated to recognise the new role.

Investigator sessions differ from worker sessions in intent, not mechanics:

- Same runtime (tmux/process)
- Same worktree mechanism
- Same lifecycle states
- **Different default behavior**: triage + issue filing, not code editing + PR creation
- **Different kanban placement**: new `investigating` column (see Dashboard section)

---

## Decision: two trigger entry points

**Entry point 1 — raw webhook event:** `workflow_run.completed` arrives, `findAffectedSessions()` returns empty, trigger engine fires. Requires the flaky guard and failure classification because the webhook fires for every failure including infrastructure and flaky tests.

**Entry point 2 — `ci_failed` session status:** `SessionStatus` already has a `ci_failed` value. When a tracked session transitions into `ci_failed`, the lifecycle manager can directly invoke the trigger engine with a confirmed failure signal. No flaky guard needed — the lifecycle has already confirmed the failure is real.

Both entry points produce the same outcome: an investigator session spawns. The distinction is only in how much pre-filtering is needed.

---

## Decision: flaky guard is mandatory for webhook entry point

Before spawning an investigator for a raw CI failure webhook, AO triggers a re-run of the failed workflow via the GitHub API and waits for the result.

| Re-run result | Action |
|---|---|
| Passes | Event was flaky. No spawn. Record in trigger log. |
| Fails again | Failure is persistent. Proceed to failure classification. |
| Re-run API fails | Treat as persistent. Proceed (conservative default). |
| Re-run times out (>10 min) | Treat as persistent. Proceed. |

Configurable via `triage.rerun_first: false` to skip for projects where CI is stable.

**Why mandatory?** CI flakiness rates in active OSS repos are high. Without the guard, a noisy test suite generates a continuous stream of investigator sessions and GitHub Issues, eroding team trust faster than it builds it.

Not required for the `ci_failed` session status entry point — the lifecycle has already confirmed persistence.

---

## Decision: failure classification before spawning

After the flaky guard confirms a persistent failure, AO classifies the failure by parsing the step name from the `workflow_run` payload.

| Step category | Examples | Action |
|---|---|---|
| `infra` | `setup`, `checkout`, `cache-restore`, `install-deps` | Skip — not a code bug |
| `code` | `build`, `test`, `lint`, `typecheck`, `test-web` | Investigate |
| `deploy` | `deploy`, `publish`, `release` | Skip by default (configurable) |

If the failing step is `infra`, AO does not spawn. It records the event in the trigger log with reason `infra_step_skipped`.

Additionally, AO fetches the first 100 lines of the step log and scans for exclusion patterns:

```yaml
triage:
  exclude_error_patterns:
    - "ECONNREFUSED"
    - "rate limit"
    - "Connection timeout"
    - "ETIMEDOUT"
```

If any pattern matches, the spawn is skipped with reason `error_pattern_excluded`.

---

## Cycle prevention

Three mechanisms together prevent infinite spawn loops:

### 1. Branch scope

Trigger rules for CI failures apply to `main`/`release` branches only by default. CI failures on PR branches are excluded. Agent-created PRs run CI on their own branch — those failures never reach the trigger engine.

```yaml
filter:
  branches: [main]
```

### 2. Label exclusion

Every GitHub Issue filed by an investigator gets the label `agent-triaged`. Every PR created by a fix session gets the label `agent-fix`. Trigger rules exclude both by default:

```yaml
filter:
  exclude_issue_labels: [agent-triaged, agent-fix]
```

### 3. Event fingerprint deduplication

Before spawning, AO checks whether an investigator was already spawned for the same event fingerprint:

```
fingerprint = hash(workflowName + headSHA)      # for CI failures
fingerprint = hash(issueNumber + labelName)     # for issue labels
```

If a matching fingerprint exists in the trigger log:
- Prior session still active → skip, no new spawn
- Prior session completed → comment on existing GitHub Issue, no duplicate spawn

### 4. Max retries per rule

```yaml
max_retries: 2
```

After a rule fires `max_retries` times for the same recurring failure pattern, it transitions to `throttled` state and notifies instead of spawning. Operator resets manually via CLI.

---

## The trigger → investigate → act lifecycle

### 1. Webhook arrives at existing route

`/api/webhooks/[...slug]/route.ts` verifies signature and parses the event. No changes to this logic.

### 2. `findAffectedSessions()` returns empty

The existing code `continue`s here. New behavior:

```typescript
const triggers = findMatchingTriggers(services.config, event, candidate.projectId);
if (triggers.length > 0) {
  await triggerEngine.evaluate(triggers, event, candidate);
}
```

### 3. Trigger engine evaluates rules

For each matching rule:
1. Check event fingerprint → skip if duplicate
2. Run flaky guard (re-run CI, await result)
3. Classify failure step → skip if infra
4. Check error pattern exclusions → skip if matched
5. Check max retries → throttle if exceeded

If all checks pass: spawn.

### 4. Investigator session spawns

`SessionManager.spawn()` is called with the rule's `spawn` config. Session metadata is written with `role: "investigator"` via the existing `writeMetadata()` mechanism. The session appears in the new `investigating` column on the dashboard.

### 5. Investigator triages

The investigator agent:
1. Reads the CI log URL from the session prompt
2. Identifies the failing step and error output
3. Locates relevant files
4. Files a GitHub Issue with: failing step, error snippet, suspected files, classification

### 6. Investigator decides

Based on configured `on_confident_fix`:

- `issue_only` (default): session transitions to `done`. Issue filed. Human picks it up.
- `pr` (opt-in): if confident it is a code regression, self-assigns the filed issue and proceeds to a fix PR as a normal worker session. Session metadata updates to `role: "worker"`.

### 7. Session cleanup

Investigator session follows the existing `cleanup → done` lifecycle. Worktree and tmux session are removed. The filed GitHub Issue and trigger log entry persist.

---

## Config shape

```yaml
triggers:
  - event: "workflow_run.completed"
    filter:
      conclusion: "failure"
      branches: [main]
      step_categories: [build, test, lint, typecheck]
      exclude_error_patterns:
        - "ECONNREFUSED"
        - "rate limit"
        - "timeout"
      exclude_issue_labels: [agent-triaged, agent-fix]
    triage:
      rerun_first: true
      min_consecutive_failures: 2
    spawn:
      agent: "claude-code"
      prompt: |
        CI workflow '{{workflow.name}}' failed on main at step '{{step.name}}'.
        Log: {{log.url}}
        Investigate the root cause. File a GitHub Issue with your findings.
    on_confident_fix: issue_only
    max_retries: 2
    enabled: true

  - event: "issues.labeled"
    filter:
      label: "agent-ready"
      exclude_issue_labels: [agent-triaged]
    spawn:
      agent: "claude-code"
      issue_id: "{{issue.number}}"
    on_confident_fix: pr
    max_retries: 5
    enabled: true
```

---

## Dashboard: new `investigating` column

The existing coding kanban has six columns: `working`, `pending`, `review`, `respond`, `merge`, `done`.

A new `investigating` column is proposed between `working` and `pending`:

```
working | investigating | pending | review | respond | merge | done
```

Investigator sessions appear here with:
- Role badge: `investigator` (distinct color from `worker`)
- The event that triggered them: `CI: build failed on main`
- Current step: `reading log` / `filing issue`
- Link to the filed GitHub Issue once created

Investigator sessions do not appear in the sidebar session list — they are transient and do not represent long-lived work items the operator needs to track across days.

---

## Trigger log

AO maintains a per-project trigger log in the existing flat-file metadata store (same `writeMetadata` / `readMetadata` mechanism used by sessions). Each entry records:

```
eventType, fingerprint, matchedRule, disposition (spawned / skipped / throttled),
skipReason (if skipped), sessionId (if spawned), issueUrl (if filed), timestamp
```

CLI commands (following existing naming conventions):
- `ao trigger list` — list configured trigger rules and their current state
- `ao trigger log [--project <id>] [--limit 20]` — show recent trigger activity
- `ao trigger test <rule-name>` — dry-run a rule against a recent or synthetic event

---

## What is in v1

- `triggers:` config schema + `TriggerRule` type definitions
- `findMatchingTriggers()` in `scm-webhooks.ts` (additive to existing route)
- Two trigger entry points: raw webhook + `ci_failed` session status
- Flaky guard (mandatory for webhook entry point, configurable off)
- Failure step classification (infra vs. code categories)
- Error pattern exclusion
- `"investigator"` as a new role value in the existing metadata role mechanism
- `investigating` column on the coding kanban
- Event fingerprint deduplication
- `agent-triaged` / `agent-fix` label tagging
- Cycle prevention: branch scope + label exclusion + fingerprint dedup + max retries
- Trigger log (per-project, using existing flat-file metadata store)
- CLI: `ao trigger list`, `ao trigger log`, `ao trigger test` (dry-run)
- GitHub SCM only (uses existing `scm-github` plugin)
- Two supported events: `workflow_run.completed` and `issues.labeled`

## What is NOT in v1

- GitLab SCM support
- Non-SCM event sources (generic HTTP webhooks, monitoring alerts)
- Cron-based triggers (separate feature)
- Dashboard UI for managing trigger rules (YAML config first)
- `on_confident_fix: pr` as the default
- Fan-out (one rule spawning multiple sessions)
- Multi-event trigger rules (fire when A AND B both occur)
- Automatic merge of agent-fix PRs

---

## Frequently asked questions

### Will every CI failure create a new session?

No. Three checks run before any spawn: the flaky guard (re-run the CI job — if it passes, it was flaky and nothing happens), the step classifier (infra-level failures like `setup` or `checkout` are skipped), and the error pattern exclusion (network errors, rate limits, and timeouts are skipped).

### What if the same workflow keeps failing every day?

The `max_retries` config caps how many times a rule fires for a recurring failure pattern. After the cap is hit, the rule enters `throttled` state and notifies instead of spawning.

### What if the investigator files a wrong or unhelpful issue?

The filed issue is labeled `agent-triaged` and can be closed by any team member. The investigator never pushes code or opens a PR unless `on_confident_fix: pr` is explicitly configured — the worst case is a noise issue on the tracker, not a bad PR.

### What does the investigator's GitHub Issue look like?

A structured issue with sections: **Trigger** (which workflow, which commit), **Failing step** (step name, log snippet), **Classification** (code regression / uncertain — with reasoning), **Suspected files**, **Recommended action**.

### Can I trigger investigations for events other than CI failures?

Yes — `issues.labeled` is the second supported event in v1. An issue labeled `agent-ready` spawns a worker session that picks up the issue as a normal `ao spawn` would.

### How does this interact with the existing `reactions:` config?

The existing `reactions:` system fires on events affecting *existing* sessions. Trigger rules fire when *no session exists yet*. They are complementary and do not overlap. A `workflow_run.completed` event first passes through `findAffectedSessions()` — if it matches an existing session, reactions handle it. Only if no session is found does the trigger engine evaluate.

### Can I preview what a trigger rule would do without it firing?

Yes: `ao trigger test <rule-name> --dry-run` evaluates the rule against a recent real event and reports what it would do — spawn, skip (with reason), or throttle — without spawning anything.

### What is the resource footprint?

One investigator session = 1 tmux session + 1 git worktree. Worktrees are lightweight (shared object store). Investigator sessions are short-lived — typically 2–5 minutes from spawn to done.

---

## Open questions (resolved)

1. **New route or extend existing?** — Extend existing. Single inbound path. Trigger evaluation is additive.
2. **Should the spawned session be a worker or a new role?** — New `investigator` role value, extending the existing metadata role mechanism.
3. **Is the flaky guard mandatory?** — Mandatory by default for webhook entry point, configurable off. Not needed for `ci_failed` entry point.
4. **Does `on_confident_fix: pr` transition the session role?** — Yes. The investigator self-assigns the filed issue and session metadata updates to `role: "worker"` before the fix phase begins.
5. **What is the deduplication unit for `issues.labeled`?** — `hash(issueNumber + labelName)`. If the same issue is labeled twice, the second event is a fresh fingerprint and spawns a new session.
