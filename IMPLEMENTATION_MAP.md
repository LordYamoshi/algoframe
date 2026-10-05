# Implementation Map

| Reliability goal | Implementation |
|---|---|
| Authoritative lifecycle | `algoframe/reliability.rs` lifecycle state machine + `algoframe_lifecycle_event` |
| Idempotent execution | deterministic execution ID |
| Persist before remote mutation | `prepare_execution` |
| Dispatch marker | `mark_dispatching` |
| Successful remote mutation | `mark_applied` |
| Unknown remote outcome | `mark_uncertain` |
| Startup reconciliation | `reconcile_startup` |
| Decision/outcome lifecycle sync | `sync_from_decisions` |
| Circuit breaker | `circuit_breaker_status` |
| Manual freeze | `reliability_trip_circuit` |
| Manual clear | `reliability_clear_circuit` |
| Failure-rate limit | execution journal lookback |
| Mutation-rate limit | 1-minute action window |
| Daily-loss limit | persisted AlgoFrame decisions |
| Market quality/anomaly limit | current market snapshot |
| Model health/fallback limit | Ultimate V5 model health |
| Main WTB/WTS mutation protection | live scraper `progress_buying` / `progress_selling` |
| Pre-pass delete protection | live scraper cleanup guard |
| Portfolio delete protection | portfolio allocation guard |
| Fake-market E2E harness | `run_fake_market_suite` + Rust unit test |
| Replay release gate | `evaluate_release_gate` |
| Walk-forward gate | Ultimate V5 advanced validation |
| Leakage gate | Ultimate V5 leakage detector |
| Stress-test gate | Ultimate V5 synthetic stress tests |
| Persisted release history | `algoframe_release_gate` |
| Operator UI | Product Polish Reliability tab |
| Global freeze indicator | AlgoFrame header status component |
