
use std::{
    cmp::Ordering,
    collections::HashMap,
    hash::{Hash, Hasher},
};

use chrono::{DateTime, Datelike, Timelike, Utc};
use rand::{
    rngs::StdRng,
    Rng,
    SeedableRng,
};
use rand_distr::{Beta, Distribution, Normal};

use super::types::*;

#[derive(Clone, Copy, Debug)]
pub struct WeightedSample {
    pub value: f64,
    pub weight: f64,
}

#[derive(Clone, Debug, Default)]
pub struct SurvivalObservation {
    pub hours: f64,
    pub event: bool,
    pub weight: f64,
}

#[derive(Clone, Debug, Default)]
pub struct BanditEvidence {
    pub successes: f64,
    pub failures: f64,
    pub trials: f64,
    pub reward_samples: Vec<WeightedSample>,
    pub fill_observations: Vec<SurvivalObservation>,
}

impl BanditEvidence {
    pub fn reward_mean(&self) -> f64 {
        robust_weighted_mean(&self.reward_samples, 0.0)
    }

    pub fn reward_std(&self) -> f64 {
        let mean = self.reward_mean();
        let total = total_weight(&self.reward_samples);
        if total <= 0.0 {
            return 0.25;
        }

        let variance = self
            .reward_samples
            .iter()
            .filter(|sample| sample.weight > 0.0 && sample.value.is_finite())
            .map(|sample| (sample.value - mean).powi(2) * sample.weight)
            .sum::<f64>()
            / total;

        variance.sqrt().max(0.02)
    }

    pub fn fill_probability(&self, hours: f64, fallback: f64) -> f64 {
        if self.fill_observations.len() < 3 {
            return fallback.clamp(0.01, 0.99);
        }

        (1.0 - kaplan_meier_survival(&self.fill_observations, hours))
            .clamp(0.01, 0.99)
    }

    pub fn median_fill_hours(&self, fallback: f64) -> f64 {
        if self.fill_observations.len() < 3 {
            return fallback;
        }

        let mut checkpoints: Vec<f64> = self
            .fill_observations
            .iter()
            .map(|observation| observation.hours.max(0.01))
            .collect();

        checkpoints.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        checkpoints.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

        checkpoints
            .into_iter()
            .find(|hours| kaplan_meier_survival(&self.fill_observations, *hours) <= 0.5)
            .unwrap_or(fallback)
    }

    pub fn expected_fill_hours(&self, fallback: f64) -> f64 {
        if self.fill_observations.len() < 3 {
            return fallback;
        }

        let max_hours = self
            .fill_observations
            .iter()
            .map(|observation| observation.hours)
            .fold(0.0_f64, f64::max)
            .max(fallback)
            .clamp(1.0, 720.0);

        let steps = 96usize;
        let dt = max_hours / steps as f64;
        let mut area = 0.0;

        for index in 0..steps {
            let t0 = index as f64 * dt;
            let t1 = (index + 1) as f64 * dt;
            let s0 = kaplan_meier_survival(&self.fill_observations, t0);
            let s1 = kaplan_meier_survival(&self.fill_observations, t1);
            area += (s0 + s1) * 0.5 * dt;
        }

        area.clamp(0.05, 720.0)
    }

    pub fn survival_prediction(&self, fallback_hours: f64) -> SurvivalPrediction {
        SurvivalPrediction {
            fill_15m: self.fill_probability(0.25, 1.0 - (-0.25 / fallback_hours.max(0.1)).exp()),
            fill_1h: self.fill_probability(1.0, 1.0 - (-1.0 / fallback_hours.max(0.1)).exp()),
            fill_6h: self.fill_probability(6.0, 1.0 - (-6.0 / fallback_hours.max(0.1)).exp()),
            fill_24h: self.fill_probability(24.0, 1.0 - (-24.0 / fallback_hours.max(0.1)).exp()),
            median_fill_hours: self.median_fill_hours(fallback_hours),
            expected_fill_hours: self.expected_fill_hours(fallback_hours),
            calibration: 0.5,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ElasticityModel {
    pub intercept: f64,
    pub price_coefficient: f64,
    pub quantity_coefficient: f64,
    pub sample_count: usize,
}

impl ElasticityModel {
    pub fn fit(records: &[(f64, f64, bool, f64)]) -> Self {
        if records.len() < 5 {
            return Self {
                intercept: -0.3,
                price_coefficient: 0.45,
                quantity_coefficient: -0.12,
                sample_count: records.len(),
            };
        }

        // Small online logistic-regression fit. Inputs are price advantage,
        // quantity and sample weight. This is deliberately tiny and bounded.
        let mut intercept = 0.0;
        let mut price_coefficient = 0.25;
        let mut quantity_coefficient = -0.05;
        let learning_rate = 0.025;

        for _ in 0..100 {
            let mut g0 = 0.0;
            let mut g1 = 0.0;
            let mut g2 = 0.0;
            let mut weight_sum = 0.0;

            for (price_advantage, quantity, filled, weight) in records {
                let z = intercept
                    + price_coefficient * price_advantage.clamp(-10.0, 10.0)
                    + quantity_coefficient * quantity.clamp(1.0, 20.0);

                let prediction = sigmoid(z);
                let target = if *filled { 1.0 } else { 0.0 };
                let error = target - prediction;
                let w = weight.max(0.01);

                g0 += error * w;
                g1 += error * price_advantage.clamp(-10.0, 10.0) * w;
                g2 += error * quantity.clamp(1.0, 20.0) * w;
                weight_sum += w;
            }

            if weight_sum <= 0.0 {
                break;
            }

            intercept += learning_rate * g0 / weight_sum;
            price_coefficient += learning_rate * g1 / weight_sum;
            quantity_coefficient += learning_rate * g2 / weight_sum;

            intercept = intercept.clamp(-5.0, 5.0);
            price_coefficient = price_coefficient.clamp(-2.0, 2.0);
            quantity_coefficient = quantity_coefficient.clamp(-2.0, 2.0);
        }

        Self {
            intercept,
            price_coefficient,
            quantity_coefficient,
            sample_count: records.len(),
        }
    }

    pub fn probability_multiplier(&self, price_advantage: f64, quantity: i64) -> f64 {
        let baseline = sigmoid(self.intercept);
        let current = sigmoid(
            self.intercept
                + self.price_coefficient * price_advantage.clamp(-10.0, 10.0)
                + self.quantity_coefficient * (quantity as f64).clamp(1.0, 20.0),
        );

        if baseline <= 0.0 {
            1.0
        } else {
            (current / baseline).clamp(0.25, 3.0)
        }
    }
}

pub fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

pub fn robust_weighted_mean(samples: &[WeightedSample], fallback: f64) -> f64 {
    if samples.is_empty() {
        return fallback;
    }

    let median = weighted_median(samples).unwrap_or(fallback);
    let mad = weighted_mad(samples).unwrap_or(0.0);

    let clip = (mad * 3.5)
        .max(median.abs() * 0.15)
        .max(1.0);

    let mut weighted_sum = 0.0;
    let mut total = 0.0;

    for sample in samples {
        if sample.weight <= 0.0 || !sample.value.is_finite() {
            continue;
        }

        let value = sample.value.clamp(median - clip, median + clip);
        weighted_sum += value * sample.weight;
        total += sample.weight;
    }

    if total > 0.0 {
        weighted_sum / total
    } else {
        fallback
    }
}

pub fn weighted_median(samples: &[WeightedSample]) -> Option<f64> {
    let mut filtered: Vec<WeightedSample> = samples
        .iter()
        .copied()
        .filter(|sample| sample.weight > 0.0 && sample.value.is_finite())
        .collect();

    if filtered.is_empty() {
        return None;
    }

    filtered.sort_by(|a, b| {
        a.value
            .partial_cmp(&b.value)
            .unwrap_or(Ordering::Equal)
    });

    let total = filtered.iter().map(|sample| sample.weight).sum::<f64>();
    let midpoint = total / 2.0;
    let mut accumulated = 0.0;

    for sample in filtered {
        accumulated += sample.weight;
        if accumulated >= midpoint {
            return Some(sample.value);
        }
    }

    None
}

pub fn weighted_mad(samples: &[WeightedSample]) -> Option<f64> {
    let median = weighted_median(samples)?;

    let deviations: Vec<WeightedSample> = samples
        .iter()
        .map(|sample| WeightedSample {
            value: (sample.value - median).abs(),
            weight: sample.weight,
        })
        .collect();

    weighted_median(&deviations)
}

pub fn total_weight(samples: &[WeightedSample]) -> f64 {
    samples
        .iter()
        .filter(|sample| sample.weight > 0.0)
        .map(|sample| sample.weight)
        .sum()
}

pub fn recency_weight(at: DateTime<Utc>, half_life_days: f64) -> f64 {
    let age_days = (Utc::now() - at).num_seconds().max(0) as f64 / 86_400.0;
    0.5_f64.powf(age_days / half_life_days.max(1.0))
}

pub fn kaplan_meier_survival(observations: &[SurvivalObservation], at_hours: f64) -> f64 {
    if observations.is_empty() || at_hours <= 0.0 {
        return 1.0;
    }

    let mut times: Vec<f64> = observations
        .iter()
        .filter(|observation| observation.hours >= 0.0 && observation.weight > 0.0)
        .map(|observation| observation.hours)
        .collect();

    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let mut survival = 1.0;

    for time in times {
        if time > at_hours {
            break;
        }

        let at_risk = observations
            .iter()
            .filter(|observation| observation.hours >= time)
            .map(|observation| observation.weight.max(0.0))
            .sum::<f64>();

        if at_risk <= 0.0 {
            continue;
        }

        let events = observations
            .iter()
            .filter(|observation| {
                observation.event && (observation.hours - time).abs() < 1e-9
            })
            .map(|observation| observation.weight.max(0.0))
            .sum::<f64>();

        survival *= (1.0 - events / at_risk).clamp(0.0, 1.0);
    }

    survival.clamp(0.0, 1.0)
}

pub fn thompson_utility(
    evidence: &BanditEvidence,
    prior: (f64, f64),
    predicted_reward: f64,
    uncertainty: f64,
    rng: &mut StdRng,
) -> f64 {
    let alpha = (prior.0 + evidence.successes).max(0.1);
    let beta = (prior.1 + evidence.failures).max(0.1);

    let fill_sample = Beta::new(alpha, beta)
        .ok()
        .map(|distribution| distribution.sample(rng))
        .unwrap_or(0.5);

    let learned_weight =
        (evidence.trials / (evidence.trials + 8.0)).clamp(0.0, 0.90);

    let learned_reward_sample = if evidence.reward_samples.is_empty() {
        predicted_reward
    } else {
        let mean = evidence.reward_mean();
        let std_error = evidence.reward_std()
            / (evidence.reward_samples.len() as f64).sqrt().max(1.0);

        Normal::new(mean, std_error.max(0.01))
            .ok()
            .map(|distribution| distribution.sample(rng))
            .unwrap_or(mean)
    };

    let reward = predicted_reward * (1.0 - learned_weight)
        + learned_reward_sample * learned_weight;

    fill_sample * reward * (1.0 + uncertainty.clamp(0.0, 1.0) * 0.05)
}

pub fn estimate_propensities<F>(
    candidate_keys: &[String],
    samples: usize,
    seed: u64,
    mut sampler: F,
) -> HashMap<String, f64>
where
    F: FnMut(&str, &mut StdRng) -> f64,
{
    if candidate_keys.is_empty() {
        return HashMap::new();
    }

    let mut counts: HashMap<String, usize> = candidate_keys
        .iter()
        .map(|key| (key.clone(), 0usize))
        .collect();

    let iterations = samples.max(16);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA17F_C0DE_9E37_79B9);

    for _ in 0..iterations {
        let mut best: Option<(&String, f64)> = None;

        for key in candidate_keys {
            let utility = sampler(key, &mut rng);
            match best {
                Some((_, current)) if utility <= current => {}
                _ => best = Some((key, utility)),
            }
        }

        if let Some((key, _)) = best {
            *counts.entry(key.clone()).or_default() += 1;
        }
    }

    let denominator = iterations as f64;
    counts
        .into_iter()
        .map(|(key, count)| (key, (count as f64 / denominator).max(1.0 / denominator)))
        .collect()
}

pub fn make_seed(parts: &[&str]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for part in parts {
        part.hash(&mut hasher);
    }
    hasher.finish()
}

pub fn stable_fraction(parts: &[&str]) -> f64 {
    (make_seed(parts) as f64) / (u64::MAX as f64)
}

pub fn normalized_reward(
    profit: f64,
    capital: f64,
    hours: f64,
    human_minutes: f64,
    human_time_value_plat_per_hour: f64,
) -> f64 {
    if capital <= 0.0 || hours <= 0.0 {
        return 0.0;
    }

    let human_cost =
        (human_minutes.max(0.0) / 60.0) * human_time_value_plat_per_hour.max(0.0);

    let net_profit = profit - human_cost;

    // Percentage return on capital per hour. Human interaction cost is converted
    // to an equivalent platinum cost before the normalization.
    (net_profit / capital) / hours.max(0.05) * 100.0
}

pub fn detect_regime(
    live_reference: f64,
    historical_reference: f64,
    week_shift_percent: f64,
    recent_mid_prices: &[f64],
) -> (MarketRegime, f64) {
    if live_reference <= 0.0 || historical_reference <= 0.0 {
        return (MarketRegime::Normal, 0.0);
    }

    let displacement =
        ((live_reference - historical_reference) / historical_reference).abs();

    let weekly = (week_shift_percent / 100.0).abs();

    let change_point = if recent_mid_prices.len() >= 6 {
        let split = recent_mid_prices.len() / 2;
        let first = mean(&recent_mid_prices[..split]).max(1.0);
        let second = mean(&recent_mid_prices[split..]);
        ((second - first) / first).abs()
    } else {
        0.0
    };

    let score = displacement.max(weekly).max(change_point);

    if score >= 0.28 {
        (MarketRegime::Shock, score)
    } else if score >= 0.12 {
        (MarketRegime::Moving, score)
    } else {
        (MarketRegime::Normal, score)
    }
}

pub fn anomaly_report(
    buy_prices: &[i64],
    sell_prices: &[i64],
    recent_mid_prices: &[f64],
    recent_churn: f64,
) -> AnomalyReport {
    let mut report = AnomalyReport::default();

    let (_, low_removed) = robust_floor_price(sell_prices);
    let (_, high_removed) = robust_ceiling_price(buy_prices);

    report.suspicious_low_listings = low_removed;
    report.suspicious_high_bids = high_removed;

    if low_removed > 0 {
        report.reasons.push(format!(
            "{} isolated low sell listing(s)",
            low_removed
        ));
    }

    if high_removed > 0 {
        report.reasons.push(format!(
            "{} isolated high buy listing(s)",
            high_removed
        ));
    }

    if recent_mid_prices.len() >= 4 {
        let current = *recent_mid_prices.last().unwrap_or(&0.0);
        let previous = recent_mid_prices[recent_mid_prices.len() - 2].max(1.0);
        let jump = ((current - previous) / previous).abs();

        if jump >= 0.20 {
            report.sudden_price_jump = true;
            report.reasons.push(format!("sudden mid-price jump {:.1}%", jump * 100.0));
        }
    }

    if recent_churn >= 0.75 {
        report.suspicious_churn = true;
        report.reasons.push("unusually high order-book churn".into());
    }

    if buy_prices.len() + sell_prices.len() < 4 {
        report.shallow_book = true;
        report.reasons.push("very shallow order book".into());
    }

    report.score = (
        low_removed as f64 * 0.15
            + high_removed as f64 * 0.15
            + if report.sudden_price_jump { 0.30 } else { 0.0 }
            + if report.suspicious_churn { 0.25 } else { 0.0 }
            + if report.shallow_book { 0.15 } else { 0.0 }
    )
    .clamp(0.0, 1.0);

    report
}

pub fn data_quality(
    buy_prices: &[i64],
    sell_prices: &[i64],
    historical_reference: f64,
    anomaly: &AnomalyReport,
    age_seconds: f64,
) -> DataQuality {
    let order_count = buy_prices.len() + sell_prices.len();

    let depth_quality = (order_count as f64 / 12.0).clamp(0.0, 1.0);
    let book_quality = if !buy_prices.is_empty() && !sell_prices.is_empty() {
        1.0
    } else if order_count > 0 {
        0.45
    } else {
        0.0
    };
    let historical_quality = if historical_reference > 0.0 { 1.0 } else { 0.35 };
    let freshness_quality = (-age_seconds.max(0.0) / 120.0).exp().clamp(0.0, 1.0);
    let anomaly_penalty = anomaly.score;

    let score = (
        historical_quality * 0.20
            + book_quality * 0.35
            + depth_quality * 0.25
            + freshness_quality * 0.20
    ) * (1.0 - anomaly_penalty * 0.65);

    let mut reasons = Vec::new();
    if depth_quality < 0.4 {
        reasons.push("limited market depth".into());
    }
    if book_quality < 0.8 {
        reasons.push("one side of order book is missing".into());
    }
    if historical_quality < 0.8 {
        reasons.push("weak historical baseline".into());
    }
    if anomaly_penalty > 0.4 {
        reasons.push("anomalous market state".into());
    }

    DataQuality {
        score: score.clamp(0.0, 1.0),
        historical_quality,
        order_book_quality: book_quality,
        depth_quality,
        freshness_quality,
        anomaly_penalty,
        reasons,
    }
}

pub fn robust_floor_price(prices: &[i64]) -> (i64, usize) {
    let mut values: Vec<i64> = prices.iter().copied().filter(|price| *price > 0).collect();
    if values.is_empty() {
        return (0, 0);
    }

    values.sort_unstable();
    let mut start = 0usize;
    let mut removed = 0usize;

    while start + 2 < values.len() && removed < 3 {
        let current = values[start];
        let next = values[start + 1];

        let gap = next - current;
        let ratio = gap as f64 / current.max(1) as f64;
        let duplicate = values[start + 1] == current;

        if !duplicate && gap >= 3 && ratio >= 0.15 {
            start += 1;
            removed += 1;
        } else {
            break;
        }
    }

    (values[start], removed)
}

pub fn robust_ceiling_price(prices: &[i64]) -> (i64, usize) {
    let mut values: Vec<i64> = prices.iter().copied().filter(|price| *price > 0).collect();
    if values.is_empty() {
        return (0, 0);
    }

    values.sort_unstable();
    let mut end = values.len();
    let mut removed = 0usize;

    while end >= 3 && removed < 3 {
        let current = values[end - 1];
        let next = values[end - 2];

        let gap = current - next;
        let ratio = gap as f64 / next.max(1) as f64;
        let duplicate = values[end - 2] == current;

        if !duplicate && gap >= 3 && ratio >= 0.15 {
            end -= 1;
            removed += 1;
        } else {
            break;
        }
    }

    (*values[..end].last().unwrap_or(&0), removed)
}

pub fn order_book_dynamics(
    recent: &[(DateTime<Utc>, i64, i64, i64, i64)],
) -> OrderBookDynamics {
    if recent.len() < 2 {
        return OrderBookDynamics::default();
    }

    let first = recent.first().unwrap();
    let last = recent.last().unwrap();

    let hours = (last.0 - first.0).num_seconds().max(1) as f64 / 3600.0;

    let first_spread = (first.2 - first.1) as f64;
    let last_spread = (last.2 - last.1) as f64;

    let total_moves = recent
        .windows(2)
        .map(|window| {
            let a = window[0];
            let b = window[1];

            ((b.1 - a.1).abs()
                + (b.2 - a.2).abs()
                + (b.3 - a.3).abs()
                + (b.4 - a.4).abs()) as f64
        })
        .sum::<f64>();

    let normalization = recent.len().saturating_sub(1).max(1) as f64;

    OrderBookDynamics {
        bid_velocity_per_hour: (last.1 - first.1) as f64 / hours,
        ask_velocity_per_hour: (last.2 - first.2) as f64 / hours,
        spread_velocity_per_hour: (last_spread - first_spread) / hours,
        buy_depth_change_per_hour: (last.3 - first.3) as f64 / hours,
        sell_depth_change_per_hour: (last.4 - first.4) as f64 / hours,
        churn_score: (total_moves / normalization / 10.0).clamp(0.0, 1.0),
        competitive_response_score: (
            ((last.1 - first.1).abs() + (last.2 - first.2).abs()) as f64
                / hours.max(0.25)
                / 8.0
        )
        .clamp(0.0, 1.0),
    }
}

pub fn forecast_prices(
    recent: &[(DateTime<Utc>, f64, f64)],
) -> Forecast {
    if recent.len() < 3 {
        return Forecast::default();
    }

    let origin = recent.first().unwrap().0;

    let x: Vec<f64> = recent
        .iter()
        .map(|(time, _, _)| (*time - origin).num_seconds() as f64 / 3600.0)
        .collect();

    let mids: Vec<f64> = recent.iter().map(|(_, mid, _)| *mid).collect();
    let spreads: Vec<f64> = recent.iter().map(|(_, _, spread)| *spread).collect();

    let (mid_intercept, mid_slope, mid_r2) = linear_regression(&x, &mids);
    let (spread_intercept, spread_slope, spread_r2) = linear_regression(&x, &spreads);

    let now_x = *x.last().unwrap_or(&0.0);

    let predict_mid = |hours: f64| {
        (mid_intercept + mid_slope * (now_x + hours)).max(0.0)
    };

    let predict_spread = |hours: f64| {
        (spread_intercept + spread_slope * (now_x + hours)).max(0.0)
    };

    let current_mid = *mids.last().unwrap_or(&0.0);
    let current_spread = *spreads.last().unwrap_or(&0.0);

    let opportunity_2h = if current_mid > 0.0 {
        ((predict_spread(2.0) - current_spread) / current_mid)
            .clamp(-1.0, 1.0)
    } else {
        0.0
    };

    Forecast {
        price_30m: predict_mid(0.5),
        price_2h: predict_mid(2.0),
        price_8h: predict_mid(8.0),
        spread_30m: predict_spread(0.5),
        opportunity_2h,
        confidence: ((mid_r2 + spread_r2) * 0.5).clamp(0.0, 1.0),
    }
}

pub fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    if x.len() != y.len() || x.len() < 2 {
        return (0.0, 0.0, 0.0);
    }

    let x_mean = mean(x);
    let y_mean = mean(y);

    let numerator = x
        .iter()
        .zip(y)
        .map(|(xv, yv)| (xv - x_mean) * (yv - y_mean))
        .sum::<f64>();

    let denominator = x
        .iter()
        .map(|xv| (xv - x_mean).powi(2))
        .sum::<f64>();

    let slope = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };

    let intercept = y_mean - slope * x_mean;

    let ss_total = y.iter().map(|yv| (yv - y_mean).powi(2)).sum::<f64>();
    let ss_residual = x
        .iter()
        .zip(y)
        .map(|(xv, yv)| {
            let predicted = intercept + slope * xv;
            (yv - predicted).powi(2)
        })
        .sum::<f64>();

    let r2 = if ss_total > 0.0 {
        (1.0 - ss_residual / ss_total).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (intercept, slope, r2)
}

pub fn market_embedding(features: &MarketFeatures) -> Vec<f64> {
    vec![
        normalize(features.spread_percent, 0.0, 1.0),
        normalize(features.liquidity, 0.0, 1.0),
        normalize(features.volatility, 0.0, 2.0),
        normalize(features.week_shift / 100.0, -1.0, 1.0),
        normalize(features.inventory_pressure, 0.0, 1.0),
        normalize(features.event_signal, -1.0, 1.0),
        normalize(features.arbitrage_signal, -1.0, 1.0),
        normalize(features.anomaly.score, 0.0, 1.0),
        normalize(features.quality.score, 0.0, 1.0),
        normalize(features.dynamics.bid_velocity_per_hour, -10.0, 10.0),
        normalize(features.dynamics.ask_velocity_per_hour, -10.0, 10.0),
        normalize(features.dynamics.spread_velocity_per_hour, -10.0, 10.0),
        normalize(features.dynamics.churn_score, 0.0, 1.0),
        normalize(features.dynamics.competitive_response_score, 0.0, 1.0),
        normalize(features.forecast.opportunity_2h, -1.0, 1.0),
        features.lifecycle.numeric(),
        features.time_hour_sin,
        features.time_hour_cos,
        features.weekday_sin,
        features.weekday_cos,
        match features.regime {
            MarketRegime::Normal => 0.0,
            MarketRegime::Moving => 0.5,
            MarketRegime::Shock => 1.0,
        },
    ]
}

pub fn learned_embedding_weights(decisions: &[DecisionRecord]) -> Vec<f64> {
    let usable: Vec<&DecisionRecord> = decisions
        .iter()
        .filter(|decision| {
            decision.actual_reward.is_some()
                && !decision.features.embedding.is_empty()
        })
        .collect();

    let dimension = usable
        .first()
        .map(|decision| decision.features.embedding.len())
        .unwrap_or(0);

    if dimension == 0 {
        return vec![];
    }

    if usable.len() < 20 {
        return vec![1.0; dimension];
    }

    let rewards: Vec<f64> = usable
        .iter()
        .map(|decision| decision.actual_reward.unwrap_or(0.0))
        .collect();

    let mut weights = Vec::with_capacity(dimension);

    for index in 0..dimension {
        let values: Vec<f64> = usable
            .iter()
            .map(|decision| {
                decision
                    .features
                    .embedding
                    .get(index)
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();

        let correlation = pearson_correlation(&values, &rewards).abs();

        // Keep every feature alive with a floor so a temporarily weak
        // correlation cannot permanently remove a useful signal. Stronger
        // features receive up to ~4x the weight of the floor.
        weights.push((0.25 + correlation * 0.75).clamp(0.25, 1.0));
    }

    let mean_weight = mean(&weights).max(0.01);
    for weight in &mut weights {
        *weight = (*weight / mean_weight).clamp(0.25, 4.0);
    }

    weights
}

pub fn weighted_cosine_similarity(
    a: &[f64],
    b: &[f64],
    weights: &[f64],
) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let use_weights = weights.len() == a.len();

    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;

    for index in 0..a.len() {
        let weight = if use_weights {
            weights[index].max(0.0)
        } else {
            1.0
        };

        dot += a[index] * b[index] * weight;
        na += a[index] * a[index] * weight;
        nb += b[index] * b[index] * weight;
    }

    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
    }
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

pub fn nearest_neighbor_reward(
    embedding: &[f64],
    history: &[(Vec<f64>, f64, f64)],
    k: usize,
    feature_weights: &[f64],
) -> (f64, f64) {
    let mut neighbors: Vec<(f64, f64, f64)> = history
        .iter()
        .map(|(candidate, reward, weight)| {
            (
                weighted_cosine_similarity(
                    embedding,
                    candidate,
                    feature_weights,
                ),
                *reward,
                *weight,
            )
        })
        .filter(|(similarity, _, weight)| *similarity > 0.0 && *weight > 0.0)
        .collect();

    neighbors.sort_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
    });
    neighbors.truncate(k.max(1));

    let mut weighted_reward = 0.0;
    let mut total = 0.0;

    for (similarity, reward, weight) in neighbors {
        let w = similarity.powi(2) * weight;
        weighted_reward += reward * w;
        total += w;
    }

    if total > 0.0 {
        (weighted_reward / total, (total / (total + 4.0)).clamp(0.0, 1.0))
    } else {
        (0.0, 0.0)
    }
}

pub fn calibration_error(predictions: &[(f64, bool)]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }

    let buckets = 10usize;
    let mut totals = vec![0usize; buckets];
    let mut predicted_sum = vec![0.0; buckets];
    let mut actual_sum = vec![0.0; buckets];

    for (prediction, actual) in predictions {
        let bucket = ((*prediction).clamp(0.0, 0.999_999) * buckets as f64) as usize;
        totals[bucket] += 1;
        predicted_sum[bucket] += prediction.clamp(0.0, 1.0);
        actual_sum[bucket] += if *actual { 1.0 } else { 0.0 };
    }

    let n = predictions.len() as f64;

    (0..buckets)
        .filter(|bucket| totals[*bucket] > 0)
        .map(|bucket| {
            let count = totals[bucket] as f64;
            let p = predicted_sum[bucket] / count;
            let a = actual_sum[bucket] / count;
            (count / n) * (p - a).abs()
        })
        .sum::<f64>()
}

pub fn brier_score(predictions: &[(f64, bool)]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }

    predictions
        .iter()
        .map(|(prediction, actual)| {
            let target = if *actual { 1.0 } else { 0.0 };
            (prediction.clamp(0.0, 1.0) - target).powi(2)
        })
        .sum::<f64>()
        / predictions.len() as f64
}

pub fn offline_policy_evaluation(
    decisions: &[DecisionRecord],
    target_policy_name: &str,
) -> OfflinePolicyEvaluation {
    let completed: Vec<&DecisionRecord> = decisions
        .iter()
        .filter(|decision| {
            matches!(
                decision.status,
                DecisionStatus::Completed
                    | DecisionStatus::PaperCompleted
                    | DecisionStatus::Expired
            )
        })
        .collect();

    if completed.is_empty() {
        return OfflinePolicyEvaluation {
            policy_name: target_policy_name.to_string(),
            reasons: vec!["no completed decisions".into()],
            ..Default::default()
        };
    }

    let mut ips_num = 0.0;
    let mut ips_den = 0.0;
    let mut direct_values = Vec::new();
    let mut dr_values = Vec::new();
    let mut realized_rewards = Vec::new();

    // q-hat is an intentionally simple context/action reward model built from
    // observed rewards. This lets the estimator remain doubly robust without
    // requiring a heavyweight offline ML dependency.
    let mut qhat: HashMap<(String, String), Vec<f64>> = HashMap::new();

    for decision in &completed {
        if let Some(reward) = decision.actual_reward {
            qhat.entry((
                decision.context_key.clone(),
                decision.chosen_action.clone(),
            ))
            .or_default()
            .push(reward);
        }
    }

    for decision in completed {
        let Some(reward) = decision.actual_reward else {
            continue;
        };

        realized_rewards.push(reward);

        let shadow_action = decision
            .shadow_actions
            .get(target_policy_name)
            .cloned()
            .unwrap_or_else(|| decision.chosen_action.clone());

        let q_target = qhat
            .get(&(decision.context_key.clone(), shadow_action.clone()))
            .map(|values| mean(values))
            .unwrap_or(0.0);

        let q_logged = qhat
            .get(&(
                decision.context_key.clone(),
                decision.chosen_action.clone(),
            ))
            .map(|values| mean(values))
            .unwrap_or(0.0);

        direct_values.push(q_target);

        if shadow_action == decision.chosen_action {
            let propensity = decision.chosen_propensity.max(0.01);
            let weight = 1.0 / propensity;

            ips_num += reward * weight;
            ips_den += weight;

            dr_values.push(q_target + weight * (reward - q_logged));
        } else {
            dr_values.push(q_target);
        }
    }

    let average_reward = mean(&realized_rewards);
    let median_reward = median(&realized_rewards);
    let failure_rate = decisions
        .iter()
        .filter(|decision| decision.actual_reward.unwrap_or(0.0) <= 0.0)
        .count() as f64
        / decisions.len().max(1) as f64;

    let direct = mean(&direct_values);
    let dr = mean(&dr_values);
    let ips = if ips_den > 0.0 { ips_num / ips_den } else { 0.0 };

    let std = std_dev(&realized_rewards);
    let margin = if realized_rewards.is_empty() {
        0.0
    } else {
        1.96 * std / (realized_rewards.len() as f64).sqrt()
    };

    OfflinePolicyEvaluation {
        policy_name: target_policy_name.to_string(),
        metrics: EvaluationMetrics {
            sample_count: realized_rewards.len(),
            average_reward,
            median_reward,
            failure_rate,
            ips_reward: ips,
            snips_reward: ips,
            doubly_robust_reward: dr,
            max_drawdown: maximum_drawdown(&dr_values),
            ..Default::default()
        },
        confidence_low: direct - margin,
        confidence_high: direct + margin,
        promotable: false,
        reasons: vec![],
    }
}

pub fn feature_importance(decisions: &[DecisionRecord]) -> Vec<FeatureImportance> {
    let rows: Vec<(&DecisionRecord, f64)> = decisions
        .iter()
        .filter_map(|decision| decision.actual_reward.map(|reward| (decision, reward)))
        .collect();

    if rows.len() < 8 {
        return vec![];
    }

    let feature_names: Vec<String> = rows[0].0.features.flat_map().keys().cloned().collect();
    let rewards: Vec<f64> = rows.iter().map(|(_, reward)| *reward).collect();

    let mut result = Vec::new();

    for feature in feature_names {
        let values: Vec<f64> = rows
            .iter()
            .map(|(decision, _)| {
                decision
                    .features
                    .flat_map()
                    .get(&feature)
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();

        let correlation = pearson_correlation(&values, &rewards);
        result.push(FeatureImportance {
            feature,
            correlation,
            importance: correlation.abs(),
            sample_count: rows.len(),
        });
    }

    result.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(Ordering::Equal)
    });

    result
}

pub fn model_health(
    decisions: &[DecisionRecord],
    max_drawdown_pct: f64,
) -> ModelHealth {
    let mut completed: Vec<&DecisionRecord> = decisions
        .iter()
        .filter(|decision| decision.actual_reward.is_some())
        .collect();

    completed.sort_by_key(|decision| decision.updated_at);

    if completed.len() < 10 {
        return ModelHealth {
            score: 0.75,
            healthy: true,
            reasons: vec!["limited learning history".into()],
            ..Default::default()
        };
    }

    let rewards: Vec<f64> = completed
        .iter()
        .filter_map(|decision| decision.actual_reward)
        .collect();

    let split = completed.len().saturating_sub(50);
    let recent = &completed[split..];

    let recent_reward = mean(
        &recent
            .iter()
            .filter_map(|decision| decision.actual_reward)
            .collect::<Vec<_>>(),
    );

    let baseline_reward = mean(&rewards);

    let recent_failure_rate = recent
        .iter()
        .filter(|decision| decision.actual_reward.unwrap_or(0.0) <= 0.0)
        .count() as f64
        / recent.len().max(1) as f64;

    let recent_prediction_errors: Vec<f64> = recent
        .iter()
        .filter_map(|decision| {
            decision.actual_profit.map(|actual| {
                (actual - decision.predicted_profit).abs()
            })
        })
        .collect();

    let recent_prediction_mae = mean(&recent_prediction_errors);

    let baseline_prediction_errors: Vec<f64> = completed
        .iter()
        .filter_map(|decision| {
            decision
                .actual_profit
                .map(|actual| (actual - decision.predicted_profit).abs())
        })
        .collect();
    let baseline_prediction_mae = mean(&baseline_prediction_errors);

    let calibration_predictions: Vec<(f64, bool)> = recent
        .iter()
        .map(|decision| {
            (
                decision.predicted_fill.fill_1h,
                decision.actual_fill_hours.unwrap_or(f64::INFINITY) <= 1.0,
            )
        })
        .collect();

    let calibration = calibration_error(&calibration_predictions);
    let drawdown = maximum_drawdown(&rewards);
    let drift_score = decision_feature_drift(&completed);

    let mut score: f64 = 1.0;
    let mut reasons = Vec::new();

    if baseline_reward > 0.0 && recent_reward < baseline_reward * 0.65 {
        score -= 0.25;
        reasons.push("recent reward materially below baseline".into());
    }

    if recent_failure_rate > 0.60 {
        score -= 0.20;
        reasons.push("recent failure rate is high".into());
    }

    if calibration > 0.20 {
        score -= 0.15;
        reasons.push("fill probabilities are poorly calibrated".into());
    }

    if baseline_prediction_mae > 0.0
        && recent_prediction_mae > baseline_prediction_mae * 1.75
        && recent_prediction_errors.len() >= 5
    {
        score -= 0.20;
        reasons.push("system-wide prediction drift detected".into());
    }

    if drawdown > max_drawdown_pct {
        score -= 0.25;
        reasons.push("model drawdown exceeded configured limit".into());
    }

    if drift_score > 0.55 {
        score -= 0.20;
        reasons.push("system-wide feature/data drift is high".into());
    } else if drift_score > 0.35 {
        score -= 0.10;
        reasons.push("moderate feature/data drift detected".into());
    }

    let open_decisions = decisions
        .iter()
        .filter(|decision| {
            matches!(
                decision.status,
                DecisionStatus::Open
                    | DecisionStatus::Partial
                    | DecisionStatus::PaperOpen
            )
        })
        .count();

    let inventory_backlog =
        open_decisions as f64 / decisions.len().max(1) as f64;

    if inventory_backlog > 0.55 {
        score -= 0.10;
        reasons.push("open-order/inventory backlog is elevated".into());
    }

    let healthy = score >= 0.55;

    ModelHealth {
        score: score.clamp(0.0, 1.0),
        healthy,
        fallback_active: !healthy,
        reasons,
        recent_reward,
        baseline_reward,
        recent_failure_rate,
        recent_prediction_mae,
        calibration_error: calibration,
        drift_score,
        inventory_backlog,
        drawdown,
    }
}

pub fn decision_feature_drift(decisions: &[&DecisionRecord]) -> f64 {
    if decisions.len() < 30 {
        return 0.0;
    }

    let split = decisions.len().saturating_sub(40);
    let recent = &decisions[split..];
    let baseline = &decisions[..split];

    if baseline.len() < 10 || recent.len() < 10 {
        return 0.0;
    }

    let Some(first) = decisions.first() else {
        return 0.0;
    };

    let feature_names: Vec<String> =
        first.features.flat_map().keys().cloned().collect();

    let mut shifts = Vec::new();

    for feature in feature_names {
        let baseline_values: Vec<f64> = baseline
            .iter()
            .map(|decision| {
                decision
                    .features
                    .flat_map()
                    .get(&feature)
                    .copied()
                    .unwrap_or(0.0)
            })
            .filter(|value| value.is_finite())
            .collect();

        let recent_values: Vec<f64> = recent
            .iter()
            .map(|decision| {
                decision
                    .features
                    .flat_map()
                    .get(&feature)
                    .copied()
                    .unwrap_or(0.0)
            })
            .filter(|value| value.is_finite())
            .collect();

        if baseline_values.len() < 5 || recent_values.len() < 5 {
            continue;
        }

        let baseline_mean = mean(&baseline_values);
        let recent_mean = mean(&recent_values);
        let baseline_std = std_dev(&baseline_values);

        let scale = baseline_std
            .max(baseline_mean.abs() * 0.10)
            .max(0.05);

        shifts.push(
            (((recent_mean - baseline_mean).abs() / scale).clamp(0.0, 4.0))
                / 4.0,
        );
    }

    if shifts.is_empty() {
        return 0.0;
    }

    shifts.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    let take = shifts.len().min(8);

    mean(&shifts[..take]).clamp(0.0, 1.0)
}

pub fn maximum_drawdown(rewards: &[f64]) -> f64 {
    if rewards.is_empty() {
        return 0.0;
    }

    let mut cumulative = 0.0;
    let mut peak = 0.0;
    let mut max_drawdown = 0.0;

    for reward in rewards {
        cumulative += reward;
        peak = peak.max(cumulative);

        if peak > 0.0 {
            max_drawdown = max_drawdown.max((peak - cumulative) / peak);
        }
    }

    max_drawdown
}

pub fn time_features(now: DateTime<Utc>) -> (f64, f64, f64, f64) {
    let hour = now.hour() as f64 + now.minute() as f64 / 60.0;
    let hour_angle = hour / 24.0 * std::f64::consts::TAU;

    let weekday = now.weekday().num_days_from_monday() as f64;
    let weekday_angle = weekday / 7.0 * std::f64::consts::TAU;

    (
        hour_angle.sin(),
        hour_angle.cos(),
        weekday_angle.sin(),
        weekday_angle.cos(),
    )
}

pub fn settings_hash(config: &UltimateConfig) -> String {
    let json = serde_json::to_string(config).unwrap_or_default();
    format!("{:016x}", make_seed(&[&json]))
}

pub fn category_from(tags: &[String], item_name: &str) -> String {
    let normalized: Vec<String> = tags.iter().map(|tag| tag.to_lowercase()).collect();

    for candidate in [
        "arcane",
        "relic",
        "mod",
        "prime",
        "weapon",
        "warframe",
        "companion",
        "syndicate",
    ] {
        if normalized.iter().any(|tag| tag.contains(candidate))
            || item_name.to_lowercase().contains(candidate)
        {
            return candidate.to_string();
        }
    }

    "other".to_string()
}

pub fn family_from_name(item_name: &str) -> String {
    let mut value = item_name.trim().to_string();

    for suffix in [
        " Set",
        " Blueprint",
        " Chassis",
        " Neuroptics",
        " Systems",
        " Barrel",
        " Receiver",
        " Stock",
        " Blade",
        " Handle",
        " Link",
        " Grip",
        " String",
        " Upper Limb",
        " Lower Limb",
        " Ornament",
        " Carapace",
        " Cerebrum",
        " Harness",
        " Wings",
    ] {
        if value.ends_with(suffix) {
            value.truncate(value.len() - suffix.len());
            break;
        }
    }

    value.trim().to_lowercase()
}

pub fn normalize(value: f64, minimum: f64, maximum: f64) -> f64 {
    if maximum <= minimum {
        return 0.0;
    }

    ((value - minimum) / (maximum - minimum))
        .clamp(-1.0, 1.0)
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut values = values.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    values[values.len() / 2]
}

pub fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let m = mean(values);
    (values
        .iter()
        .map(|value| (value - m).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

pub fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 3 {
        return 0.0;
    }

    let ma = mean(a);
    let mb = mean(b);

    let numerator = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>();

    let da = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
    let db = b.iter().map(|y| (y - mb).powi(2)).sum::<f64>().sqrt();

    if da <= 0.0 || db <= 0.0 {
        0.0
    } else {
        (numerator / (da * db)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn survival_declines_after_events() {
        let observations = vec![
            SurvivalObservation { hours: 1.0, event: true, weight: 1.0 },
            SurvivalObservation { hours: 2.0, event: true, weight: 1.0 },
            SurvivalObservation { hours: 3.0, event: false, weight: 1.0 },
        ];

        assert!(kaplan_meier_survival(&observations, 2.0) < 1.0);
    }

    #[test]
    fn robust_mean_ignores_extreme_outlier() {
        let samples = vec![
            WeightedSample { value: 10.0, weight: 1.0 },
            WeightedSample { value: 11.0, weight: 1.0 },
            WeightedSample { value: 12.0, weight: 1.0 },
            WeightedSample { value: 1000.0, weight: 1.0 },
        ];

        assert!(robust_weighted_mean(&samples, 0.0) < 100.0);
    }

    #[test]
    fn fast_trade_has_better_reward() {
        let fast = normalized_reward(15.0, 50.0, 1.0, 1.0, 0.0);
        let slow = normalized_reward(30.0, 100.0, 24.0, 1.0, 0.0);

        assert!(fast > slow);
    }

    #[test]
    fn regime_detects_large_move() {
        let (regime, _) = detect_regime(140.0, 100.0, 0.0, &[100.0, 101.0, 130.0, 140.0]);
        assert_eq!(regime, MarketRegime::Shock);
    }

    #[test]
    fn family_normalizer_groups_prime_parts() {
        assert_eq!(
            family_from_name("Xaku Prime Systems"),
            family_from_name("Xaku Prime Set")
        );
    }
}
