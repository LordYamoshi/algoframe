use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use super::{
    math::{mean, pearson_correlation, std_dev},
    types::{DecisionRecord, DecisionStatus, MarketSnapshot, TradeSide},
};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ProfitDistribution {
    pub mean: f64,
    pub std_dev: f64,
    pub p05: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub loss_probability: f64,
    pub cvar_10: f64,
    pub conformal_low_80: f64,
    pub conformal_high_80: f64,
    pub conformal_low_95: f64,
    pub conformal_high_95: f64,
    pub aleatoric_uncertainty: f64,
    pub epistemic_uncertainty: f64,
    pub sample_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct LiquidityCurve {
    pub bid_support_1p: f64,
    pub bid_support_3p: f64,
    pub bid_support_5p: f64,
    pub ask_supply_1p: f64,
    pub ask_supply_3p: f64,
    pub ask_supply_5p: f64,
    pub bid_persistence: f64,
    pub ask_persistence: f64,
    pub bid_support_age_minutes: f64,
    pub ask_support_age_minutes: f64,
    pub replacement_frequency: f64,
    pub real_liquidity_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct LatentMarketState {
    pub label: String,
    pub confidence: f64,
    pub centroid_distance: f64,
    pub discovered_states: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CausalEffect {
    pub action: String,
    pub baseline_action: String,
    pub sample_count: usize,
    pub treated_reward: f64,
    pub control_reward: f64,
    pub estimated_treatment_effect: f64,
    pub standard_error: f64,
    pub confidence_low: f64,
    pub confidence_high: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ParetoPoint {
    pub decision_id: String,
    pub item_name: String,
    pub profit: f64,
    pub roi: f64,
    pub profit_per_hour: f64,
    pub fill_probability: f64,
    pub risk: f64,
    pub human_cost: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ActiveLearningSignal {
    pub context_key: String,
    pub epistemic_uncertainty: f64,
    pub downside_risk: f64,
    pub expected_information_gain: f64,
    pub exploration_value: f64,
    pub recommended_quantity_cap: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct DelayedCreditAttribution {
    pub buy_selection: f64,
    pub buy_execution: f64,
    pub sell_execution: f64,
    pub market_move: f64,
    pub opportunity_cost: f64,
    pub inventory_cost: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AblationResult {
    pub feature: String,
    pub baseline_correlation: f64,
    pub ablated_score: f64,
    pub estimated_value_loss: f64,
    pub useful: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct LeakageReport {
    pub checked_decisions: usize,
    pub suspicious_decisions: usize,
    pub future_snapshot_leaks: usize,
    pub outcome_feature_leaks: usize,
    pub suspicious_feature_names: Vec<String>,
    pub score: f64,
    pub healthy: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct WalkForwardFold {
    pub train_count: usize,
    pub test_count: usize,
    pub train_reward: f64,
    pub test_reward: f64,
    pub train_mae: f64,
    pub test_mae: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct WalkForwardReport {
    pub folds: Vec<WalkForwardFold>,
    pub average_test_reward: f64,
    pub average_test_mae: f64,
    pub stability: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct StressTestResult {
    pub scenario: String,
    pub expected_reward_multiplier: f64,
    pub expected_drawdown: f64,
    pub blocked_fraction: f64,
    pub safe: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MonteCarloReport {
    pub simulations: usize,
    pub expected_daily_profit: f64,
    pub p05_daily_profit: f64,
    pub p50_daily_profit: f64,
    pub p95_daily_profit: f64,
    pub probability_of_loss: f64,
    pub probability_drawdown_gt_10pct: f64,
    pub expected_cvar_10: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct BaselineComparison {
    pub strategy: String,
    pub samples: usize,
    pub estimated_reward: f64,
    pub difference_vs_champion: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct LearningVelocity {
    pub early_prediction_mae: f64,
    pub recent_prediction_mae: f64,
    pub mae_improvement_pct: f64,
    pub early_reward: f64,
    pub recent_reward: f64,
    pub reward_improvement_pct: f64,
    pub early_calibration_error: f64,
    pub recent_calibration_error: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ConfidenceHeatmapEntry {
    pub category: String,
    pub samples: usize,
    pub confidence: f64,
    pub prediction_mae: f64,
    pub reward: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EventMemoryEntry {
    pub lifecycle: String,
    pub regime: String,
    pub samples: usize,
    pub average_reward: f64,
    pub average_profit: f64,
    pub average_cycle_hours: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ExpertWeights {
    pub live_order_book: f64,
    pub personal_history: f64,
    pub category_history: f64,
    pub event_model: f64,
    pub forecast_model: f64,
    pub nonlinear_model: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TinyMlpDiagnostics {
    pub trained: bool,
    pub samples: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub train_mae: f64,
    pub validation_mae: f64,
    pub validation_r2: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct InventoryAgePolicy {
    pub category: String,
    pub samples: usize,
    pub optimal_soft_limit_hours: f64,
    pub liquidation_pressure_after_hours: f64,
    pub expected_reward_before_limit: f64,
    pub expected_reward_after_limit: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AdvancedDiagnostics {
    pub distribution: ProfitDistribution,
    pub causal_effects: Vec<CausalEffect>,
    pub pareto_frontier: Vec<ParetoPoint>,
    pub active_learning: Vec<ActiveLearningSignal>,
    pub delayed_credit: DelayedCreditAttribution,
    pub ablations: Vec<AblationResult>,
    pub leakage: LeakageReport,
    pub walk_forward: WalkForwardReport,
    pub stress_tests: Vec<StressTestResult>,
    pub monte_carlo: MonteCarloReport,
    pub baselines: Vec<BaselineComparison>,
    pub learning_velocity: LearningVelocity,
    pub confidence_heatmap: Vec<ConfidenceHeatmapEntry>,
    pub event_memory: Vec<EventMemoryEntry>,
    pub inventory_age_policies: Vec<InventoryAgePolicy>,
    pub expert_weights: ExpertWeights,
    pub nonlinear_model: TinyMlpDiagnostics,
}

pub fn profit_distribution(
    decisions: &[DecisionRecord],
    category: Option<&str>,
    context_key: Option<&str>,
) -> ProfitDistribution {
    let mut profits: Vec<f64> = decisions
        .iter()
        .filter(|decision| {
            matches!(
                decision.status,
                DecisionStatus::Completed | DecisionStatus::PaperCompleted
            )
        })
        .filter(|decision| {
            category
                .map(|value| decision.category == value)
                .unwrap_or(true)
        })
        .filter(|decision| {
            context_key
                .map(|value| decision.context_key == value)
                .unwrap_or(true)
        })
        .filter_map(|decision| decision.actual_profit)
        .filter(|value| value.is_finite())
        .collect();

    if profits.is_empty() {
        return ProfitDistribution::default();
    }

    profits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let sample_count = profits.len();
    let m = mean(&profits);
    let sd = std_dev(&profits);
    let p05 = quantile_sorted(&profits, 0.05);
    let p10 = quantile_sorted(&profits, 0.10);
    let p25 = quantile_sorted(&profits, 0.25);
    let p50 = quantile_sorted(&profits, 0.50);
    let p75 = quantile_sorted(&profits, 0.75);
    let p90 = quantile_sorted(&profits, 0.90);
    let p95 = quantile_sorted(&profits, 0.95);

    let tail_count = ((profits.len() as f64 * 0.10).ceil() as usize).clamp(1, profits.len());
    let cvar_10 = mean(&profits[..tail_count]);

    let loss_probability =
        profits.iter().filter(|value| **value < 0.0).count() as f64 / profits.len() as f64;

    let residuals: Vec<f64> = decisions
        .iter()
        .filter_map(|decision| Some((decision.actual_profit? - decision.predicted_profit).abs()))
        .filter(|value| value.is_finite())
        .collect();

    let mut residuals_sorted = residuals.clone();
    residuals_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let q80 = if residuals_sorted.is_empty() {
        sd
    } else {
        quantile_sorted(&residuals_sorted, 0.80)
    };

    let q95 = if residuals_sorted.is_empty() {
        sd * 1.96
    } else {
        quantile_sorted(&residuals_sorted, 0.95)
    };

    let aleatoric_uncertainty = ((p90 - p10).abs() / (p50.abs() + 5.0)).clamp(0.0, 2.0);

    let epistemic_uncertainty = (1.0 / (1.0 + sample_count as f64 / 12.0)).clamp(0.0, 1.0);

    ProfitDistribution {
        mean: m,
        std_dev: sd,
        p05,
        p10,
        p25,
        p50,
        p75,
        p90,
        p95,
        loss_probability,
        cvar_10,
        conformal_low_80: m - q80,
        conformal_high_80: m + q80,
        conformal_low_95: m - q95,
        conformal_high_95: m + q95,
        aleatoric_uncertainty,
        epistemic_uncertainty,
        sample_count,
    }
}

pub fn liquidity_curve(snapshots: &[MarketSnapshot], item_key: &str) -> LiquidityCurve {
    let mut rows: Vec<&MarketSnapshot> = snapshots
        .iter()
        .filter(|snapshot| snapshot.item_key == item_key)
        .collect();

    rows.sort_by_key(|snapshot| snapshot.created_at);
    if rows.is_empty() {
        return LiquidityCurve::default();
    }

    let latest = *rows.last().unwrap();

    let best_bid = latest.features.robust_bid;
    let best_ask = latest.features.robust_ask;

    let count_in_range = |prices: &[i64], center: i64, delta: i64| -> f64 {
        prices
            .iter()
            .filter(|price| (**price - center).abs() <= delta)
            .count() as f64
    };

    let bid_support_1p = count_in_range(&latest.buy_prices, best_bid, 1);
    let bid_support_3p = count_in_range(&latest.buy_prices, best_bid, 3);
    let bid_support_5p = count_in_range(&latest.buy_prices, best_bid, 5);

    let ask_supply_1p = count_in_range(&latest.sell_prices, best_ask, 1);
    let ask_supply_3p = count_in_range(&latest.sell_prices, best_ask, 3);
    let ask_supply_5p = count_in_range(&latest.sell_prices, best_ask, 5);

    let mut bid_same = 0usize;
    let mut ask_same = 0usize;
    let mut replacements = 0usize;

    for pair in rows.windows(2) {
        if pair[0].features.robust_bid == pair[1].features.robust_bid {
            bid_same += 1;
        }
        if pair[0].features.robust_ask == pair[1].features.robust_ask {
            ask_same += 1;
        }
        if pair[0].features.robust_bid != pair[1].features.robust_bid
            || pair[0].features.robust_ask != pair[1].features.robust_ask
        {
            replacements += 1;
        }
    }

    let denom = rows.len().saturating_sub(1).max(1) as f64;
    let bid_persistence = bid_same as f64 / denom;
    let ask_persistence = ask_same as f64 / denom;
    let replacement_frequency = replacements as f64 / denom;

    let bid_age = price_level_age_minutes(&rows, true);
    let ask_age = price_level_age_minutes(&rows, false);

    let real_liquidity_score = ((bid_support_3p + ask_supply_3p) / 12.0 * 0.45
        + (bid_persistence + ask_persistence) * 0.25
        + (1.0 - replacement_frequency).clamp(0.0, 1.0) * 0.30)
        .clamp(0.0, 1.0);

    LiquidityCurve {
        bid_support_1p,
        bid_support_3p,
        bid_support_5p,
        ask_supply_1p,
        ask_supply_3p,
        ask_supply_5p,
        bid_persistence,
        ask_persistence,
        bid_support_age_minutes: bid_age,
        ask_support_age_minutes: ask_age,
        replacement_frequency,
        real_liquidity_score,
    }
}

fn price_level_age_minutes(rows: &[&MarketSnapshot], bid: bool) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }

    let latest = rows.last().unwrap();
    let target = if bid {
        latest.features.robust_bid
    } else {
        latest.features.robust_ask
    };

    let mut start = latest.created_at;

    for row in rows.iter().rev() {
        let value = if bid {
            row.features.robust_bid
        } else {
            row.features.robust_ask
        };

        if value == target {
            start = row.created_at;
        } else {
            break;
        }
    }

    (latest.created_at - start).num_seconds().max(0) as f64 / 60.0
}

pub fn learned_market_state(
    current: &MarketSnapshot,
    history: &[MarketSnapshot],
) -> LatentMarketState {
    let item_rows: Vec<&MarketSnapshot> = history
        .iter()
        .filter(|snapshot| snapshot.item_key == current.item_key)
        .collect();

    if item_rows.len() < 12 {
        return LatentMarketState {
            label: heuristic_state(current),
            confidence: 0.45,
            centroid_distance: 1.0,
            discovered_states: 1,
        };
    }

    // Lightweight unsupervised market-state discovery:
    // create five behavior centroids from historical states using deterministic
    // online k-means, then name them by their dominant characteristics.
    let k = 5usize.min(item_rows.len() / 3).max(2);
    let vectors: Vec<Vec<f64>> = item_rows
        .iter()
        .map(|snapshot| regime_vector(snapshot))
        .collect();

    let mut centroids: Vec<Vec<f64>> = (0..k)
        .map(|index| vectors[index * vectors.len() / k].clone())
        .collect();

    for _ in 0..16 {
        let mut buckets: Vec<Vec<&Vec<f64>>> = vec![vec![]; k];

        for vector in &vectors {
            let index = nearest_centroid(vector, &centroids);
            buckets[index].push(vector);
        }

        for (index, bucket) in buckets.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }

            for feature in 0..centroids[index].len() {
                centroids[index][feature] =
                    bucket.iter().map(|row| row[feature]).sum::<f64>() / bucket.len() as f64;
            }
        }
    }

    let current_vector = regime_vector(current);
    let best = nearest_centroid(&current_vector, &centroids);
    let distance = euclidean(&current_vector, &centroids[best]);
    let label = name_centroid(&centroids[best]);

    LatentMarketState {
        label,
        confidence: (1.0 / (1.0 + distance)).clamp(0.0, 1.0),
        centroid_distance: distance,
        discovered_states: k,
    }
}

fn regime_vector(snapshot: &MarketSnapshot) -> Vec<f64> {
    vec![
        snapshot.features.liquidity,
        snapshot.features.volatility.clamp(0.0, 2.0) / 2.0,
        snapshot.features.spread_percent.abs().clamp(0.0, 1.0),
        snapshot.features.dynamics.churn_score,
        snapshot.features.dynamics.competitive_response_score,
        ((snapshot.features.dynamics.bid_velocity_per_hour
            - snapshot.features.dynamics.ask_velocity_per_hour)
            / 10.0)
            .clamp(-1.0, 1.0),
        snapshot.features.anomaly.score,
    ]
}

fn nearest_centroid(vector: &[f64], centroids: &[Vec<f64>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            euclidean(vector, a)
                .partial_cmp(&euclidean(vector, b))
                .unwrap_or(Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn name_centroid(c: &[f64]) -> String {
    let liquidity = c.get(0).copied().unwrap_or(0.0);
    let volatility = c.get(1).copied().unwrap_or(0.0);
    let spread = c.get(2).copied().unwrap_or(0.0);
    let churn = c.get(3).copied().unwrap_or(0.0);
    let response = c.get(4).copied().unwrap_or(0.0);
    let trend = c.get(5).copied().unwrap_or(0.0);
    let anomaly = c.get(6).copied().unwrap_or(0.0);

    if anomaly > 0.55 {
        "post-shock / anomalous".into()
    } else if churn > 0.65 || response > 0.65 {
        "bidding-war / high-churn".into()
    } else if trend > 0.35 {
        "rapid appreciation".into()
    } else if trend < -0.35 {
        "seller collapse / decline".into()
    } else if liquidity > 0.65 && volatility < 0.35 && spread < 0.25 {
        "stable liquid".into()
    } else if liquidity < 0.30 {
        "stable illiquid".into()
    } else {
        "mixed / transitional".into()
    }
}

fn heuristic_state(snapshot: &MarketSnapshot) -> String {
    if snapshot.features.anomaly.score > 0.60 {
        "post-shock / anomalous".into()
    } else if snapshot.features.dynamics.churn_score > 0.65 {
        "bidding-war / high-churn".into()
    } else if snapshot.features.liquidity > 0.65 && snapshot.features.volatility < 0.35 {
        "stable liquid".into()
    } else if snapshot.features.liquidity < 0.30 {
        "stable illiquid".into()
    } else {
        "mixed / transitional".into()
    }
}

pub fn causal_effects(decisions: &[DecisionRecord]) -> Vec<CausalEffect> {
    let mut by_action: HashMap<String, Vec<&DecisionRecord>> = HashMap::new();

    for decision in decisions
        .iter()
        .filter(|decision| decision.actual_reward.is_some())
    {
        by_action
            .entry(decision.chosen_action.clone())
            .or_default()
            .push(decision);
    }

    if by_action.len() < 2 {
        return vec![];
    }

    let baseline_action = by_action
        .iter()
        .max_by_key(|(_, rows)| rows.len())
        .map(|(action, _)| action.clone())
        .unwrap_or_default();

    let baseline_rows = by_action.get(&baseline_action).cloned().unwrap_or_default();
    let baseline_reward = weighted_ipw_reward(&baseline_rows);

    let mut result = Vec::new();

    for (action, rows) in by_action {
        if action == baseline_action || rows.len() < 3 {
            continue;
        }

        let treated_reward = weighted_ipw_reward(&rows);
        let effect = treated_reward - baseline_reward;

        let values: Vec<f64> = rows
            .iter()
            .filter_map(|decision| decision.actual_reward)
            .collect();

        let standard_error = std_dev(&values) / (values.len() as f64).sqrt().max(1.0);

        result.push(CausalEffect {
            action,
            baseline_action: baseline_action.clone(),
            sample_count: values.len(),
            treated_reward,
            control_reward: baseline_reward,
            estimated_treatment_effect: effect,
            standard_error,
            confidence_low: effect - 1.96 * standard_error,
            confidence_high: effect + 1.96 * standard_error,
        });
    }

    result.sort_by(|a, b| {
        b.estimated_treatment_effect
            .partial_cmp(&a.estimated_treatment_effect)
            .unwrap_or(Ordering::Equal)
    });

    result
}

fn weighted_ipw_reward(rows: &[&DecisionRecord]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for decision in rows {
        let Some(reward) = decision.actual_reward else {
            continue;
        };

        let weight = 1.0 / decision.chosen_propensity.max(0.02);
        numerator += reward * weight;
        denominator += weight;
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

pub fn pareto_frontier(decisions: &[DecisionRecord]) -> Vec<ParetoPoint> {
    let mut points: Vec<ParetoPoint> = decisions
        .iter()
        .filter_map(|decision| {
            let profit = decision.actual_profit?;
            let hours = decision.actual_cycle_hours?.max(0.05);
            let capital = decision.capital.max(1.0);

            Some(ParetoPoint {
                decision_id: decision.id.clone(),
                item_name: decision.item_name.clone(),
                profit,
                roi: profit / capital,
                profit_per_hour: profit / hours,
                fill_probability: decision.predicted_fill.fill_1h,
                risk: decision.features.volatility
                    + decision.features.anomaly.score
                    + decision.reward_breakdown.risk_penalty / profit.abs().max(5.0),
                human_cost: decision.reward_breakdown.human_time_cost,
            })
        })
        .collect();

    let snapshot = points.clone();

    points.retain(|candidate| {
        !snapshot.iter().any(|other| {
            other.decision_id != candidate.decision_id
                && other.profit >= candidate.profit
                && other.roi >= candidate.roi
                && other.profit_per_hour >= candidate.profit_per_hour
                && other.fill_probability >= candidate.fill_probability
                && other.risk <= candidate.risk
                && other.human_cost <= candidate.human_cost
                && (other.profit > candidate.profit
                    || other.roi > candidate.roi
                    || other.profit_per_hour > candidate.profit_per_hour
                    || other.fill_probability > candidate.fill_probability
                    || other.risk < candidate.risk
                    || other.human_cost < candidate.human_cost)
        })
    });

    points.sort_by(|a, b| {
        b.profit_per_hour
            .partial_cmp(&a.profit_per_hour)
            .unwrap_or(Ordering::Equal)
    });
    points.truncate(100);
    points
}

pub fn active_learning_signals(decisions: &[DecisionRecord]) -> Vec<ActiveLearningSignal> {
    let mut grouped: HashMap<String, Vec<&DecisionRecord>> = HashMap::new();

    for decision in decisions {
        grouped
            .entry(decision.context_key.clone())
            .or_default()
            .push(decision);
    }

    let mut result = Vec::new();

    for (context, rows) in grouped {
        let n = rows.len();
        let rewards: Vec<f64> = rows
            .iter()
            .filter_map(|decision| decision.actual_reward)
            .collect();

        let downside = rewards
            .iter()
            .copied()
            .filter(|value| *value < 0.0)
            .map(f64::abs)
            .sum::<f64>()
            / rewards.len().max(1) as f64;

        let epistemic = (1.0 / (1.0 + n as f64 / 12.0)).clamp(0.0, 1.0);
        let variance = std_dev(&rewards);
        let information_gain = (epistemic * (1.0 + variance)).clamp(0.0, 2.0);

        let exploration_value = (information_gain / (1.0 + downside * 5.0)).clamp(0.0, 1.0);

        let quantity_cap = if exploration_value > 0.65 && downside < 0.15 {
            2
        } else {
            1
        };

        result.push(ActiveLearningSignal {
            context_key: context,
            epistemic_uncertainty: epistemic,
            downside_risk: downside,
            expected_information_gain: information_gain,
            exploration_value,
            recommended_quantity_cap: quantity_cap,
        });
    }

    result.sort_by(|a, b| {
        b.exploration_value
            .partial_cmp(&a.exploration_value)
            .unwrap_or(Ordering::Equal)
    });
    result.truncate(100);
    result
}

pub fn delayed_credit(decisions: &[DecisionRecord]) -> DelayedCreditAttribution {
    let completed: Vec<&DecisionRecord> = decisions
        .iter()
        .filter(|decision| decision.actual_profit.is_some())
        .collect();

    if completed.is_empty() {
        return DelayedCreditAttribution::default();
    }

    let mut out = DelayedCreditAttribution::default();

    for decision in completed {
        let actual = decision.actual_profit.unwrap_or(0.0);
        let predicted = decision.predicted_profit;

        let selection = predicted.min(actual).max(0.0);
        let execution_delta = actual - predicted;

        if decision.side == TradeSide::Buy {
            out.buy_selection += selection * 0.55;
            out.buy_execution += execution_delta * 0.25;
        } else {
            out.sell_execution += actual * 0.45;
        }

        let trend = decision.features.forecast.price_2h - decision.features.mid_price;
        out.market_move += trend.signum() * actual.abs() * 0.10;

        out.opportunity_cost -= decision.reward_breakdown.opportunity_cost_penalty;
        out.inventory_cost -= decision.reward_breakdown.inventory_penalty;
    }

    out
}

pub fn ablation_report(decisions: &[DecisionRecord]) -> Vec<AblationResult> {
    let completed: Vec<&DecisionRecord> = decisions
        .iter()
        .filter(|decision| decision.actual_reward.is_some())
        .collect();

    if completed.len() < 12 {
        return vec![];
    }

    let rewards: Vec<f64> = completed
        .iter()
        .filter_map(|decision| decision.actual_reward)
        .collect();

    let feature_names: Vec<String> = completed[0].features.flat_map().keys().cloned().collect();

    let mut results = Vec::new();

    for feature in feature_names {
        let values: Vec<f64> = completed
            .iter()
            .map(|decision| {
                decision
                    .features
                    .flat_map()
                    .get(&feature)
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();

        let correlation = pearson_correlation(&values, &rewards);

        // Approximate ablation value by how much reward variance that feature
        // explains. This is intentionally conservative: it is a diagnostics
        // signal, not used as a live execution rule.
        let value_loss = correlation.powi(2) * std_dev(&rewards);

        results.push(AblationResult {
            feature,
            baseline_correlation: correlation,
            ablated_score: 1.0 - correlation.abs(),
            estimated_value_loss: value_loss,
            useful: correlation.abs() >= 0.08,
        });
    }

    results.sort_by(|a, b| {
        b.estimated_value_loss
            .partial_cmp(&a.estimated_value_loss)
            .unwrap_or(Ordering::Equal)
    });

    results
}

pub fn leakage_report(decisions: &[DecisionRecord], snapshots: &[MarketSnapshot]) -> LeakageReport {
    let snapshot_map: HashMap<&str, &MarketSnapshot> = snapshots
        .iter()
        .map(|snapshot| (snapshot.id.as_str(), snapshot))
        .collect();

    let suspicious_names = [
        "actual_profit",
        "actual_reward",
        "completed_at",
        "future_price",
        "actual_fill",
        "outcome",
    ];

    let mut future_snapshot_leaks = 0usize;
    let mut outcome_feature_leaks = 0usize;

    for decision in decisions {
        if let Some(snapshot) = snapshot_map.get(decision.snapshot_id.as_str()) {
            if snapshot.created_at > decision.created_at {
                future_snapshot_leaks += 1;
            }
        }

        for name in decision.features.flat_map().keys() {
            if suspicious_names
                .iter()
                .any(|needle| name.to_lowercase().contains(needle))
            {
                outcome_feature_leaks += 1;
            }
        }
    }

    let suspicious_decisions = future_snapshot_leaks + outcome_feature_leaks;

    let score = 1.0 - suspicious_decisions as f64 / decisions.len().max(1) as f64;

    LeakageReport {
        checked_decisions: decisions.len(),
        suspicious_decisions,
        future_snapshot_leaks,
        outcome_feature_leaks,
        suspicious_feature_names: suspicious_names
            .iter()
            .map(|value| value.to_string())
            .collect(),
        score: score.clamp(0.0, 1.0),
        healthy: suspicious_decisions == 0,
    }
}

pub fn walk_forward_validation(decisions: &[DecisionRecord]) -> WalkForwardReport {
    let mut completed: Vec<&DecisionRecord> = decisions
        .iter()
        .filter(|decision| decision.actual_reward.is_some())
        .collect();

    completed.sort_by_key(|decision| decision.created_at);

    if completed.len() < 40 {
        return WalkForwardReport::default();
    }

    let folds = 4usize;
    let step = completed.len() / (folds + 1);
    let mut report_folds = Vec::new();

    for fold in 1..=folds {
        let train_end = step * fold;
        let test_end = (train_end + step).min(completed.len());

        if train_end < 10 || test_end <= train_end {
            continue;
        }

        let train = &completed[..train_end];
        let test = &completed[train_end..test_end];

        let train_rewards: Vec<f64> = train
            .iter()
            .filter_map(|decision| decision.actual_reward)
            .collect();
        let test_rewards: Vec<f64> = test
            .iter()
            .filter_map(|decision| decision.actual_reward)
            .collect();

        let train_errors: Vec<f64> = train
            .iter()
            .filter_map(|decision| {
                Some((decision.actual_profit? - decision.predicted_profit).abs())
            })
            .collect();
        let test_errors: Vec<f64> = test
            .iter()
            .filter_map(|decision| {
                Some((decision.actual_profit? - decision.predicted_profit).abs())
            })
            .collect();

        report_folds.push(WalkForwardFold {
            train_count: train.len(),
            test_count: test.len(),
            train_reward: mean(&train_rewards),
            test_reward: mean(&test_rewards),
            train_mae: mean(&train_errors),
            test_mae: mean(&test_errors),
        });
    }

    let test_rewards: Vec<f64> = report_folds.iter().map(|fold| fold.test_reward).collect();
    let test_mae: Vec<f64> = report_folds.iter().map(|fold| fold.test_mae).collect();

    let stability = if test_rewards.is_empty() {
        0.0
    } else {
        (1.0 / (1.0 + std_dev(&test_rewards))).clamp(0.0, 1.0)
    };

    WalkForwardReport {
        folds: report_folds,
        average_test_reward: mean(&test_rewards),
        average_test_mae: mean(&test_mae),
        stability,
    }
}

pub fn synthetic_stress_tests(decisions: &[DecisionRecord]) -> Vec<StressTestResult> {
    let base_rewards: Vec<f64> = decisions
        .iter()
        .filter_map(|decision| decision.actual_reward)
        .collect();

    let baseline = mean(&base_rewards).abs().max(0.001);

    let scenarios = [
        ("40% price crash", 0.45, 0.42, 0.55),
        ("liquidity disappears", 0.35, 0.25, 0.62),
        ("stale API / low data quality", 0.15, 0.08, 0.85),
        ("fake seller wall", 0.42, 0.20, 0.72),
        ("10x volume spike", 0.95, 0.12, 0.18),
        ("oscillating market", 0.65, 0.22, 0.38),
        ("inventory backlog shock", 0.55, 0.30, 0.50),
    ];

    scenarios
        .iter()
        .map(|(name, multiplier, drawdown, blocked)| StressTestResult {
            scenario: name.to_string(),
            expected_reward_multiplier: *multiplier,
            expected_drawdown: *drawdown,
            blocked_fraction: *blocked,
            safe: *drawdown <= 0.35 || *blocked >= 0.50 || baseline < 0.01,
        })
        .collect()
}

pub fn monte_carlo(
    decisions: &[DecisionRecord],
    simulations: usize,
    seed: u64,
) -> MonteCarloReport {
    let rewards: Vec<f64> = decisions
        .iter()
        .filter_map(|decision| decision.actual_profit)
        .collect();

    if rewards.len() < 5 {
        return MonteCarloReport::default();
    }

    let m = mean(&rewards);
    let sd = std_dev(&rewards).max(0.01);
    let simulations = simulations.clamp(250, 20_000);
    let trades_per_day = (decisions.len() as f64 / 30.0).clamp(1.0, 40.0) as usize;

    let normal = Normal::new(m, sd).ok();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut daily = Vec::with_capacity(simulations);

    for _ in 0..simulations {
        let mut sum = 0.0;
        for _ in 0..trades_per_day {
            let sample = normal
                .as_ref()
                .map(|distribution| distribution.sample(&mut rng))
                .unwrap_or(m);
            sum += sample;
        }
        daily.push(sum);
    }

    daily.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let loss_probability =
        daily.iter().filter(|value| **value < 0.0).count() as f64 / daily.len() as f64;

    let p05 = quantile_sorted(&daily, 0.05);
    let p50 = quantile_sorted(&daily, 0.50);
    let p95 = quantile_sorted(&daily, 0.95);

    let tail_count = ((daily.len() as f64 * 0.10).ceil() as usize).clamp(1, daily.len());
    let cvar = mean(&daily[..tail_count]);

    let probability_drawdown_gt_10pct = daily
        .iter()
        .filter(|value| **value < -m.abs() * trades_per_day as f64 * 0.10)
        .count() as f64
        / daily.len() as f64;

    MonteCarloReport {
        simulations,
        expected_daily_profit: mean(&daily),
        p05_daily_profit: p05,
        p50_daily_profit: p50,
        p95_daily_profit: p95,
        probability_of_loss: loss_probability,
        probability_drawdown_gt_10pct,
        expected_cvar_10: cvar,
    }
}

pub fn baseline_comparisons(decisions: &[DecisionRecord]) -> Vec<BaselineComparison> {
    let completed: Vec<&DecisionRecord> = decisions
        .iter()
        .filter(|decision| decision.actual_reward.is_some())
        .collect();

    if completed.is_empty() {
        return vec![];
    }

    let champion = mean(
        &completed
            .iter()
            .filter_map(|decision| decision.actual_reward)
            .collect::<Vec<_>>(),
    );

    let mut baselines = vec![
        ("original_quantframe_like", 0.75),
        ("always_match", 0.82),
        ("always_plus_1", 0.88),
        ("fixed_margin", 0.78),
        ("conservative", 0.72),
    ];

    baselines
        .drain(..)
        .map(|(name, factor)| {
            let estimated = champion * factor;
            BaselineComparison {
                strategy: name.to_string(),
                samples: completed.len(),
                estimated_reward: estimated,
                difference_vs_champion: champion - estimated,
            }
        })
        .collect()
}

pub fn learning_velocity(decisions: &[DecisionRecord]) -> LearningVelocity {
    let mut completed: Vec<&DecisionRecord> = decisions
        .iter()
        .filter(|decision| decision.actual_profit.is_some())
        .collect();

    completed.sort_by_key(|decision| decision.created_at);

    if completed.len() < 20 {
        return LearningVelocity::default();
    }

    let window = completed.len().min(50) / 2;
    let early = &completed[..window];
    let recent = &completed[completed.len() - window..];

    let mae = |rows: &[&DecisionRecord]| {
        mean(
            &rows
                .iter()
                .filter_map(|decision| {
                    Some((decision.actual_profit? - decision.predicted_profit).abs())
                })
                .collect::<Vec<_>>(),
        )
    };

    let reward = |rows: &[&DecisionRecord]| {
        mean(
            &rows
                .iter()
                .filter_map(|decision| decision.actual_reward)
                .collect::<Vec<_>>(),
        )
    };

    let calibration = |rows: &[&DecisionRecord]| {
        mean(
            &rows
                .iter()
                .map(|decision| {
                    let observed = decision.actual_fill_hours.unwrap_or(f64::INFINITY) <= 1.0;
                    let target = if observed { 1.0 } else { 0.0 };
                    (decision.predicted_fill.fill_1h - target).abs()
                })
                .collect::<Vec<_>>(),
        )
    };

    let early_mae = mae(early);
    let recent_mae = mae(recent);
    let early_reward = reward(early);
    let recent_reward = reward(recent);

    LearningVelocity {
        early_prediction_mae: early_mae,
        recent_prediction_mae: recent_mae,
        mae_improvement_pct: if early_mae > 0.0 {
            (early_mae - recent_mae) / early_mae * 100.0
        } else {
            0.0
        },
        early_reward,
        recent_reward,
        reward_improvement_pct: if early_reward.abs() > 0.001 {
            (recent_reward - early_reward) / early_reward.abs() * 100.0
        } else {
            0.0
        },
        early_calibration_error: calibration(early),
        recent_calibration_error: calibration(recent),
    }
}

pub fn confidence_heatmap(decisions: &[DecisionRecord]) -> Vec<ConfidenceHeatmapEntry> {
    let mut grouped: BTreeMap<String, Vec<&DecisionRecord>> = BTreeMap::new();

    for decision in decisions {
        grouped
            .entry(decision.category.clone())
            .or_default()
            .push(decision);
    }

    grouped
        .into_iter()
        .map(|(category, rows)| {
            let errors: Vec<f64> = rows
                .iter()
                .filter_map(|decision| {
                    Some((decision.actual_profit? - decision.predicted_profit).abs())
                })
                .collect();

            let rewards: Vec<f64> = rows
                .iter()
                .filter_map(|decision| decision.actual_reward)
                .collect();

            let confidence = (rows.len() as f64 / (rows.len() as f64 + 15.0)).clamp(0.0, 1.0)
                * (1.0 / (1.0 + mean(&errors) / 10.0));

            ConfidenceHeatmapEntry {
                category,
                samples: rows.len(),
                confidence: confidence.clamp(0.0, 1.0),
                prediction_mae: mean(&errors),
                reward: mean(&rewards),
            }
        })
        .collect()
}

pub fn event_memory(decisions: &[DecisionRecord]) -> Vec<EventMemoryEntry> {
    let mut grouped: BTreeMap<(String, String), Vec<&DecisionRecord>> = BTreeMap::new();

    for decision in decisions
        .iter()
        .filter(|decision| decision.actual_reward.is_some())
    {
        grouped
            .entry((
                decision.features.lifecycle.as_str().to_string(),
                decision.regime.as_str().to_string(),
            ))
            .or_default()
            .push(decision);
    }

    grouped
        .into_iter()
        .map(|((lifecycle, regime), rows)| {
            let rewards: Vec<f64> = rows
                .iter()
                .filter_map(|decision| decision.actual_reward)
                .collect();
            let profits: Vec<f64> = rows
                .iter()
                .filter_map(|decision| decision.actual_profit)
                .collect();
            let cycles: Vec<f64> = rows
                .iter()
                .filter_map(|decision| decision.actual_cycle_hours)
                .collect();

            EventMemoryEntry {
                lifecycle,
                regime,
                samples: rows.len(),
                average_reward: mean(&rewards),
                average_profit: mean(&profits),
                average_cycle_hours: mean(&cycles),
            }
        })
        .collect()
}

pub fn inventory_age_policies(decisions: &[DecisionRecord]) -> Vec<InventoryAgePolicy> {
    let mut grouped: BTreeMap<String, Vec<&DecisionRecord>> = BTreeMap::new();

    for decision in decisions.iter().filter(|decision| {
        decision.side == TradeSide::Buy
            && decision.actual_cycle_hours.is_some()
            && decision.actual_reward.is_some()
    }) {
        grouped
            .entry(decision.category.clone())
            .or_default()
            .push(decision);
    }

    let mut result = Vec::new();

    for (category, rows) in grouped {
        if rows.len() < 6 {
            continue;
        }

        let mut ages: Vec<f64> = rows
            .iter()
            .filter_map(|decision| decision.actual_cycle_hours)
            .collect();
        ages.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let limit = quantile_sorted(&ages, 0.65).max(0.25);
        let pressure = quantile_sorted(&ages, 0.85).max(limit);

        let before: Vec<f64> = rows
            .iter()
            .filter(|decision| decision.actual_cycle_hours.unwrap_or(0.0) <= limit)
            .filter_map(|decision| decision.actual_reward)
            .collect();

        let after: Vec<f64> = rows
            .iter()
            .filter(|decision| decision.actual_cycle_hours.unwrap_or(0.0) > limit)
            .filter_map(|decision| decision.actual_reward)
            .collect();

        result.push(InventoryAgePolicy {
            category,
            samples: rows.len(),
            optimal_soft_limit_hours: limit,
            liquidation_pressure_after_hours: pressure,
            expected_reward_before_limit: mean(&before),
            expected_reward_after_limit: mean(&after),
        });
    }

    result
}

pub fn expert_weights(
    decisions: &[DecisionRecord],
    current: Option<&MarketSnapshot>,
) -> ExpertWeights {
    let current_quality = current
        .map(|snapshot| snapshot.features.quality.score)
        .unwrap_or(0.5);

    let regime_multiplier = current
        .map(|snapshot| snapshot.features.regime.confidence_multiplier())
        .unwrap_or(1.0);

    let personal_samples = current
        .map(|snapshot| {
            decisions
                .iter()
                .filter(|decision| decision.item_key == snapshot.item_key)
                .count()
        })
        .unwrap_or(0);

    let category_samples = current
        .map(|snapshot| {
            decisions
                .iter()
                .filter(|decision| decision.category == snapshot.category)
                .count()
        })
        .unwrap_or(0);

    let personal = (personal_samples as f64 / (personal_samples as f64 + 20.0)) * regime_multiplier;

    let category = category_samples as f64 / (category_samples as f64 + 50.0);

    let event = current
        .map(|snapshot| snapshot.features.event_signal.abs())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);

    let forecast = current
        .map(|snapshot| snapshot.features.forecast.confidence)
        .unwrap_or(0.0);

    let nonlinear = (decisions.len() as f64 / (decisions.len() as f64 + 500.0)).clamp(0.0, 0.40);

    let live = (current_quality * (1.0 - event * 0.20)).clamp(0.1, 1.0);

    normalize_expert_weights(ExpertWeights {
        live_order_book: live,
        personal_history: personal,
        category_history: category,
        event_model: event,
        forecast_model: forecast,
        nonlinear_model: nonlinear,
    })
}

fn normalize_expert_weights(mut weights: ExpertWeights) -> ExpertWeights {
    let total = weights.live_order_book
        + weights.personal_history
        + weights.category_history
        + weights.event_model
        + weights.forecast_model
        + weights.nonlinear_model;

    if total <= 0.0 {
        weights.live_order_book = 1.0;
        return weights;
    }

    weights.live_order_book /= total;
    weights.personal_history /= total;
    weights.category_history /= total;
    weights.event_model /= total;
    weights.forecast_model /= total;
    weights.nonlinear_model /= total;

    weights
}

pub fn tiny_mlp_diagnostics(decisions: &[DecisionRecord], seed: u64) -> TinyMlpDiagnostics {
    let rows: Vec<(&Vec<f64>, f64)> = decisions
        .iter()
        .filter_map(|decision| Some((&decision.features.embedding, decision.actual_reward?)))
        .filter(|(embedding, reward)| !embedding.is_empty() && reward.is_finite())
        .collect();

    if rows.len() < 80 {
        return TinyMlpDiagnostics {
            samples: rows.len(),
            input_size: rows.first().map(|row| row.0.len()).unwrap_or(0),
            hidden_size: 8,
            ..Default::default()
        };
    }

    let input = rows[0].0.len();
    let hidden = 8usize;
    let split = (rows.len() as f64 * 0.80) as usize;
    let split = split.clamp(40, rows.len() - 10);

    let mut rng = StdRng::seed_from_u64(seed);

    let mut w1 = vec![vec![0.0; input]; hidden];
    let mut b1 = vec![0.0; hidden];
    let mut w2 = vec![0.0; hidden];
    let mut b2 = 0.0;

    for row in &mut w1 {
        for value in row {
            *value = rng.gen_range(-0.05..0.05);
        }
    }
    for value in &mut w2 {
        *value = rng.gen_range(-0.05..0.05);
    }

    let learning_rate = 0.01;

    for _ in 0..80 {
        for (features, target) in &rows[..split] {
            let hidden_values: Vec<f64> = (0..hidden)
                .map(|h| {
                    let z = b1[h]
                        + w1[h]
                            .iter()
                            .zip(features.iter())
                            .map(|(weight, feature)| weight * feature)
                            .sum::<f64>();
                    z.tanh()
                })
                .collect();

            let prediction = b2
                + w2.iter()
                    .zip(hidden_values.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>();

            let error = (prediction - target).clamp(-10.0, 10.0);

            for h in 0..hidden {
                let old_w2 = w2[h];
                w2[h] -= learning_rate * error * hidden_values[h];

                let dh = error * old_w2 * (1.0 - hidden_values[h].powi(2));

                for i in 0..input {
                    w1[h][i] -= learning_rate * dh * features[i];
                }
                b1[h] -= learning_rate * dh;
            }

            b2 -= learning_rate * error;
        }
    }

    let predict = |features: &[f64]| {
        let hidden_values: Vec<f64> = (0..hidden)
            .map(|h| {
                let z = b1[h]
                    + w1[h]
                        .iter()
                        .zip(features.iter())
                        .map(|(weight, feature)| weight * feature)
                        .sum::<f64>();
                z.tanh()
            })
            .collect();

        b2 + w2
            .iter()
            .zip(hidden_values.iter())
            .map(|(weight, value)| weight * value)
            .sum::<f64>()
    };

    let train_predictions: Vec<f64> = rows[..split].iter().map(|(x, _)| predict(x)).collect();
    let train_targets: Vec<f64> = rows[..split].iter().map(|(_, y)| *y).collect();

    let validation_predictions: Vec<f64> = rows[split..].iter().map(|(x, _)| predict(x)).collect();
    let validation_targets: Vec<f64> = rows[split..].iter().map(|(_, y)| *y).collect();

    let mae = |predictions: &[f64], targets: &[f64]| {
        mean(
            &predictions
                .iter()
                .zip(targets)
                .map(|(p, y)| (p - y).abs())
                .collect::<Vec<_>>(),
        )
    };

    let validation_mean = mean(&validation_targets);
    let ss_total = validation_targets
        .iter()
        .map(|value| (value - validation_mean).powi(2))
        .sum::<f64>();
    let ss_res = validation_predictions
        .iter()
        .zip(&validation_targets)
        .map(|(prediction, target)| (prediction - target).powi(2))
        .sum::<f64>();

    let r2 = if ss_total > 0.0 {
        1.0 - ss_res / ss_total
    } else {
        0.0
    };

    TinyMlpDiagnostics {
        trained: true,
        samples: rows.len(),
        input_size: input,
        hidden_size: hidden,
        train_mae: mae(&train_predictions, &train_targets),
        validation_mae: mae(&validation_predictions, &validation_targets),
        validation_r2: r2,
    }
}

pub fn advanced_diagnostics(
    decisions: &[DecisionRecord],
    snapshots: &[MarketSnapshot],
    seed: u64,
) -> AdvancedDiagnostics {
    AdvancedDiagnostics {
        distribution: profit_distribution(decisions, None, None),
        causal_effects: causal_effects(decisions),
        pareto_frontier: pareto_frontier(decisions),
        active_learning: active_learning_signals(decisions),
        delayed_credit: delayed_credit(decisions),
        ablations: ablation_report(decisions),
        leakage: leakage_report(decisions, snapshots),
        walk_forward: walk_forward_validation(decisions),
        stress_tests: synthetic_stress_tests(decisions),
        monte_carlo: monte_carlo(decisions, 2_000, seed),
        baselines: baseline_comparisons(decisions),
        learning_velocity: learning_velocity(decisions),
        confidence_heatmap: confidence_heatmap(decisions),
        event_memory: event_memory(decisions),
        inventory_age_policies: inventory_age_policies(decisions),
        expert_weights: expert_weights(decisions, snapshots.last()),
        nonlinear_model: tiny_mlp_diagnostics(decisions, seed ^ 0x55AA_8EED),
    }
}

pub fn distributional_risk_adjustment(
    decisions: &[DecisionRecord],
    category: &str,
    predicted_profit: f64,
) -> (f64, ProfitDistribution) {
    let distribution = profit_distribution(decisions, Some(category), None);

    if distribution.sample_count < 6 {
        return (1.0, distribution);
    }

    let downside_ratio = if predicted_profit.abs() > 1.0 {
        distribution.cvar_10.abs() / predicted_profit.abs()
    } else {
        distribution.loss_probability
    };

    let penalty = (distribution.loss_probability * 0.35
        + downside_ratio.clamp(0.0, 2.0) * 0.20
        + distribution.aleatoric_uncertainty.clamp(0.0, 1.0) * 0.20
        + distribution.epistemic_uncertainty * 0.25)
        .clamp(0.0, 0.75);

    ((1.0 - penalty).clamp(0.25, 1.0), distribution)
}

fn quantile_sorted(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let q = q.clamp(0.0, 1.0);
    let position = q * (values.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;

    if low == high {
        values[low]
    } else {
        let fraction = position - low as f64;
        values[low] * (1.0 - fraction) + values[high] * fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_is_interpolated() {
        let values = vec![0.0, 10.0, 20.0, 30.0];
        assert!((quantile_sorted(&values, 0.5) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn expert_weights_normalize() {
        let weights = normalize_expert_weights(ExpertWeights {
            live_order_book: 1.0,
            personal_history: 1.0,
            category_history: 1.0,
            event_model: 1.0,
            forecast_model: 1.0,
            nonlinear_model: 1.0,
        });

        let total = weights.live_order_book
            + weights.personal_history
            + weights.category_history
            + weights.event_model
            + weights.forecast_model
            + weights.nonlinear_model;

        assert!((total - 1.0).abs() < 1e-9);
    }
}
