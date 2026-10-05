# AlgoFrame Ultimate V4 — Feature Matrix

This file maps the major features discussed during the AlgoFrame design conversation to their V4 implementation.

| Requested feature | V4 implementation |
|---|---|
| Self-improving algorithm | Contextual Thompson-Sampling policy updated from outcomes |
| Genuine ML | Online contextual bandit + learned elasticity + survival models |
| Learn actual profit | Completed transaction outcomes |
| Learn ROI | Realized profit/capital |
| Learn WTB fill time | Censored survival observations |
| Learn WTS sell time | Censored survival observations |
| Learn full cycle time | FIFO purchase -> later-sale matching |
| Learn from unsold inventory | Inventory pressure + open-age profile |
| Learn from failures | Expired/cancelled/censored outcome evidence |
| Learn from SKIPs | Rejected decisions are persisted |
| Prediction error learning | Actual vs predicted profit |
| Recency decay | Configurable historical half-life |
| Robust outlier handling | Weighted median/MAD methods |
| Similar-item transfer | Category/global hierarchical priors |
| Market-state transfer | Embeddings + nearest-neighbour reward |
| Dynamic WTB price | Continuous candidate search |
| Dynamic WTS price | Continuous candidate search |
| Quantity learning | Price + quantity joint action space |
| Price elasticity | Small learned logistic fill model |
| Partial fills | Filled-quantity/fill-ratio tracking |
| Liquidity | Explicit feature |
| Volatility | Explicit feature |
| Spread/depth | Explicit features |
| Order-book velocity | Bid/ask/depth/spread velocity |
| Order churn | Churn score |
| Competitive response | Competitor-response feature |
| Time of day/week | Cyclic temporal features |
| Fill probability | 15m / 1h / 6h / 24h survival probabilities |
| Calibration | Brier score + expected calibration error |
| Market regime | Normal / Moving / Shock |
| Change-point detection | Recent-vs-earlier price displacement |
| System-wide drift | Feature/data drift in model health |
| Market anomaly detection | Isolated bids/asks, jump/churn/shallow-book risk |
| Data quality | Historical/book/depth/freshness/anomaly score |
| Manipulation protection | Anomaly score feeds guardrails and reward penalty |
| Event-aware trading | Warframe worldstate/news/trader signals |
| Prime Resurgence awareness | Event parser |
| Vault/unvault awareness | Event parser |
| Baro awareness | Trader parser |
| Market recorder | SQLite market snapshots |
| High-resolution retention | Raw recent snapshots |
| Downsampling | raw -> 5m -> 30m -> 1h |
| Digital twin | Paper resolution + snapshot replay |
| Counterfactual replay | Recorded future-snapshot alternative simulation |
| Propensity logging | Per-action selection probabilities |
| IPS evaluation | Offline estimator |
| SNIPS evaluation | Offline estimator |
| Doubly robust evaluation | Offline estimator |
| Shadow mode | Conservative/aggressive/fast/max-reward policies |
| Champion/challenger | Logged policy role + offline comparison |
| Automatic promotion | Evidence-gated challenger exposure increase |
| Automatic rollback | Model health activates conservative fallback |
| Model health | Reward/failure/MAE/calibration/drift/drawdown |
| Hyperparameter adaptation | Recency/exploration tuning from error/health |
| Feature discovery | Reward correlation/importance report |
| Explainability | Structured positive/negative/guardrail/confidence reasons |
| Profit attribution | Market/buy/sell/portfolio/arbitrage/event/exploration/time |
| Per-item risk cap | Portfolio risk engine |
| Category risk cap | Portfolio risk engine |
| High-volatility budget | Portfolio risk engine |
| Experimental budget | Portfolio risk engine |
| Cash reserve | Dynamic portfolio reserve |
| Opportunity cost | Forecast signal + dynamic reserve |
| Position sizing by uncertainty | Joint quantity search + risk guardrails |
| Cold-start policy | Quantity-one limit + hierarchical priors |
| Human trade capacity | Daily interaction cap |
| Human time cost | Reward penalty |
| Growth mode | Operating objective |
| Balanced mode | Operating objective |
| Conservative mode | Stricter guardrails |
| Liquid mode | Turnover-biased fill requirements |
| Paper mode | No live WFM mutations |
| Self-feedback protection | Own orders excluded before market snapshot |
| Set-vs-parts arbitrage | Family/set assembly scan |
| Arcane ranking arbitrage | Rank aggregation scan |
| Relic expected value | Public relic drop data + refinement support |
| General relationship graph | Configurable graph edges |
| Lifecycle tracking | Discovered -> ... -> sold/failed/expired |
| Deterministic replay | Snapshot/model/schema/settings/seed stored |
| Schema versioning | Feature/reward/policy/settings versions |
| Selective forgetting | Forget item / forget before date |
| Full reset | Reset learning, optionally retain snapshots |
| Dataset export | Tauri export command |
| Alerts | Model fallback/promotion/degradation alerts |
| Learning dashboard | React/Mantine `Learning` page |
