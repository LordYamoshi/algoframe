# AlgoFrame Ultimate V5 — Advanced Intelligence Coverage

V5 layers the next intelligence stack on top of Ultimate V4. It is designed to improve statistical reliability, downside-risk awareness, market-state understanding, and evidence-based self-improvement.

## Live decision systems

- [x] Distributional profit modelling instead of mean-only predictions
- [x] P05/P10/P25/P50/P75/P90/P95 outcome estimates
- [x] Loss probability
- [x] CVaR / expected worst-tail loss
- [x] Conformal 80% and 95% prediction intervals
- [x] Aleatoric uncertainty estimate
- [x] Epistemic uncertainty estimate
- [x] CVaR/downside-risk penalty in live candidate reward
- [x] Uncertainty-aware position sizing
- [x] Active-learning risk cap for uncertain experiments
- [x] Order-book persistence / real-liquidity correction
- [x] Price-level support age
- [x] Replacement-frequency / churn correction
- [x] Learned latent market-state discovery via lightweight clustering
- [x] Learned-state escalation into Moving/Shock regimes
- [x] Mixture-of-experts gating diagnostics
- [x] Existing hierarchical item/family/category/global learning retained
- [x] Existing market-state embedding generalization retained
- [x] Existing continuous price + quantity optimization retained

## Causal and counterfactual evaluation

- [x] Propensity logging retained
- [x] IPW-style causal action-effect estimates
- [x] Confidence intervals for estimated action effects
- [x] Existing IPS/SNIPS evaluation retained
- [x] Existing doubly-robust policy evaluation retained
- [x] Existing snapshot-based counterfactual replay retained
- [x] Learned SKIP baseline retained
- [x] Shadow/champion/challenger framework retained

## Validation and model science

- [x] Walk-forward temporal validation
- [x] Leakage checks
- [x] Automatic ablation diagnostics
- [x] Feature-value estimates
- [x] Synthetic stress tests
- [x] Monte Carlo daily outcome simulation
- [x] Baseline strategy comparison
- [x] Learning-velocity metrics
- [x] Confidence heatmap by category
- [x] Tiny nonlinear MLP expert diagnostics
- [x] Event/regime memory report
- [x] Model-health / rollback system retained
- [x] Drift and calibration checks retained

## Portfolio and risk

- [x] Pareto-frontier diagnostics across profit / ROI / profit-hour / fill / risk / human cost
- [x] Tail-risk reporting
- [x] Monte Carlo loss probability
- [x] Monte Carlo drawdown probability
- [x] Inventory-age policy learning
- [x] Existing capital / category / family / volatility / experimentation budgets retained
- [x] Existing dynamic cash reserve retained
- [x] Existing opportunity-cost model retained
- [x] Existing hard execution guardrails retained

## Market microstructure

- [x] Whole-book near-touch support/supply curves
- [x] Bid persistence
- [x] Ask persistence
- [x] Support age
- [x] Replacement frequency
- [x] Real-liquidity score
- [x] Existing order-book velocity retained
- [x] Existing depth/churn/competitive-response features retained
- [x] Existing anomaly/manipulation detection retained
- [x] Existing data-quality scoring retained
- [x] Existing event intelligence retained

## Warframe-specific intelligence retained from V4

- [x] Prime set-vs-parts assembly
- [x] Arcane rank aggregation
- [x] Relic expected value / refinement awareness
- [x] Relationship/conversion graph
- [x] Prime Resurgence / Baro / world-state event inputs
- [x] Inventory pressure
- [x] Human trade interaction cost
- [x] Item lifecycle memory

## Inspector additions

The `Advanced Intelligence` tab shows:

- distributional profit / CVaR
- conformal intervals
- uncertainty decomposition
- Monte Carlo portfolio outcomes
- learning velocity
- leakage status
- walk-forward validation
- nonlinear-model quality
- mixture-of-experts weights
- causal action effects
- category confidence heatmap
- baseline comparisons
- stress tests
- inventory-age policies

## Important limitation

V5 deliberately keeps the nonlinear model as an auxiliary expert/diagnostic until enough real data exists. It does not allow a small, poorly trained neural model to override hard risk controls or dominate execution. The online bandit, survival model, order-book model and distributional risk layer remain the primary live decision stack.
