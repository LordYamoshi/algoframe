# AlgoFrame Reliability V1

Reliability V1 is the operational-safety layer for AlgoFrame Ultimate V5.3 + Product Polish V1.

It implements the five reliability upgrades prioritized after the Product Polish milestone:

1. authoritative trading lifecycle state machine
2. crash-safe / idempotent WFM execution
3. live-execution circuit breakers
4. deterministic fake-market end-to-end reliability tests
5. replay-based release gates for champion/challenger promotion

## 1. Authoritative lifecycle state machine

AlgoFrame now records immutable lifecycle transitions for managed WTB/WTS decisions:

```text
Discovered
→ Evaluated
→ Approved
→ Prepared
→ Dispatching
→ Active
→ Partial
→ Purchased
→ Selling
→ Sold
```

Terminal/exception states include:

```text
Rejected
Cancelled
Superseded
Expired
Failed
Unknown
PaperSimulated
```

Transitions are stored in `algoframe_lifecycle_event`.

The learner's existing decision status remains intact; Reliability V1 synchronizes it into the execution lifecycle rather than replacing the ML dataset.

## 2. Crash-safe / idempotent execution

Before AlgoFrame mutates a live WFM order it persists an execution journal entry.

```text
decision
→ PREPARED persisted to SQLite
→ DISPATCHING persisted
→ WFM mutation
→ APPLIED persisted
```

The execution ID is deterministic for:

```text
decision + side + operation + price + quantity
```

so the same exact mutation is not blindly sent twice.

On startup, unresolved `Prepared`, `Dispatching`, or `Unknown` executions are reconciled against the current WFM order cache.

If AlgoFrame can prove that the target order exists with the expected decision ID, price and quantity, the execution becomes:

```text
Reconciled
```

If it cannot prove the remote state, it becomes:

```text
Unknown
```

and the circuit breaker freezes new live execution instead of guessing whether it is safe to retry.

Deletes use desired-state reconciliation: if the target WFM order is absent, the delete is treated as reconciled.

## 3. Circuit breakers

Live WTB/WTS execution can be frozen by:

- manual operator freeze
- model health below the reliability floor
- model fallback mode
- poor market-data quality
- extreme anomaly score
- too many recent execution failures
- excessive order-mutation rate
- unresolved/unknown remote execution state
- daily realized-loss threshold

Paper mode continues to be safe and does not perform live WFM mutations.

The breaker also protects the live-scraper pre-pass and portfolio deletion pass from deleting managed orders while execution is frozen.

The Reliability tab exposes:

- current breaker state
- reasons
- recent failures/actions
- unknown executions
- daily realized loss
- manual Freeze / Clear controls

## 4. Deterministic fake-market E2E suite

Reliability V1 includes a pure deterministic test harness covering:

- normal buy → fill → sell lifecycle
- partial fill before completion
- rejected opportunity
- crash after dispatch → Unknown instead of duplicate retry
- Paper decision that never enters live execution

The suite is available both as Rust tests and from the AlgoFrame Reliability UI.

## 5. Replay-based release gate

Future challenger policies cannot be promoted only because their average reward looks better.

Before automatic champion/challenger promotion, the release gate checks:

- minimum sample count
- doubly-robust / realized reward uplift
- failure-rate delta
- candidate maximum drawdown
- non-overlapping confidence evidence
- walk-forward stability
- leakage detector
- synthetic stress-test pass fraction

Release-gate reports are persisted in `algoframe_release_gate`.

Manual evaluation is available from the Reliability tab.

## New SQLite tables

Migration `m20261005_000002_create_algoframe_reliability.rs` adds:

```text
algoframe_execution_journal
algoframe_lifecycle_event
algoframe_circuit_event
algoframe_release_gate
```

Reliability configuration is stored in the existing `algoframe_setting` table.

## UI

Product Polish's command center gains a new:

```text
Reliability
```

tab with:

- circuit-breaker status/control
- deterministic fake-market test runner
- replay release-gate report
- crash-safe execution journal

The global header also changes to:

```text
EXECUTION FROZEN
```

when the circuit breaker is tripped.

## Install

This package expects:

- Ultimate V5.3 installed and compiling
- Product Polish V1 installed

Extract into the repository and run:

```powershell
.\Apply-AlgoFrameReliabilityV1.ps1 -RepoRoot "."
```

Then:

```powershell
.\Validate-AlgoFrameReliabilityV1.ps1 -RepoRoot "."
```

The validator automatically formats Rust first, then runs:

```text
cargo test --manifest-path src-tauri/Cargo.toml algoframe
cargo check --manifest-path src-tauri/Cargo.toml
pnpm build
```

Finally:

```powershell
pnpm tauri dev
```

## First run

Keep AlgoFrame in Paper mode first.

The new reliability migration is applied on application startup, so the Reliability tab should be tested after `pnpm tauri dev` has successfully initialized the database.

## Rollback

The installer backs up every source file it touches to:

```text
.algoframe-reliability-v1-backup
```

and checkpoints the current SQLite database as:

```text
quantframeV2.sqlite.pre-reliability
```

If installation itself fails, touched source files are restored automatically.

## Important scope

Reliability V1 journals and reconciles AlgoFrame-managed WTB/WTS mutations.

It also circuit-breaker-protects the main live-scraper cleanup/portfolio deletion paths. Wishlist and syndicate workflows remain their existing QuantFrame workflows and are not yet represented as AlgoFrame ML lifecycle decisions.


## Manual resolution of unknown executions

If an execution is `Unknown`, AlgoFrame does **not** retry it automatically.

The Reliability tab requires the operator to verify the current WFM state and choose one of:

```text
Verified applied
Safe to retry
Not applied
```

This is intentional. An uncertain remote mutation is exactly the situation where automatic retry can create duplicate or contradictory orders.

## Configurable safety thresholds

The Reliability tab exposes the main hard execution limits:

- minimum model health
- maximum anomaly score
- failures before freeze
- maximum actions per minute
- daily realized-loss limit
- prepared-execution timeout
- release-gate minimum samples
- release-gate required reward uplift

These controls are separate from the learner's prediction settings.
