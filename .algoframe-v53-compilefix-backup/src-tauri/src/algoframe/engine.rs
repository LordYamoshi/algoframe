use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
};

use chrono::{DateTime, Utc};
use entity::{
    enums::TransactionType,
    transaction::{Model as TransactionModel, TransactionPaginationQueryDto},
};
use rand::{rngs::StdRng, SeedableRng};
use service::{sea_orm::DatabaseConnection, TransactionQuery};
use utils::{Error, Properties as WfmProperties, SubType};
use uuid::Uuid;
use wf_market::{
    enums::OrderType,
    types::{Order, OrderList},
};

use crate::cache::types::{CacheTradableItem, ItemPriceInfo};

use super::{
    advanced::*,
    events::refresh_event_intelligence,
    market_data::{normalized_refinement, parse_relic_identity, relic_drop_tables, RelicDropTable},
    math::*,
    store::LearningStore,
    types::*,
};

const MAX_DECISIONS_IN_MEMORY: usize = 25_000;
const MAX_SNAPSHOTS_IN_MEMORY: usize = 8_000;
const EXPIRED_ORDER_GRACE_HOURS: f64 = 2.0;

#[derive(Clone, Debug, Default)]
struct ItemProfile {
    purchase_units: i64,
    sold_units: i64,
    matched_units: i64,
    open_units: i64,
    trade_count: usize,

    avg_profit: f64,
    avg_roi: f64,
    avg_hold_hours: f64,
    avg_sale_price: f64,
    avg_buy_fill_hours: f64,
    avg_sell_fill_hours: f64,
    avg_open_age_hours: f64,

    prediction_bias: f64,
    prediction_mae: f64,
    prediction_accuracy: f64,

    sell_through: f64,
    profit_stability: f64,
    confidence: f64,
}

#[derive(Clone, Debug)]
struct PurchaseLot {
    remaining: i64,
    unit_price: f64,
    at: DateTime<Utc>,
    decision_id: Option<String>,
    predicted_profit: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct ProfileAccumulator {
    purchase_units: i64,
    sold_units: i64,
    matched_units: i64,
    open_units: i64,
    trade_count: usize,

    profits: Vec<WeightedSample>,
    rois: Vec<WeightedSample>,
    hold_hours: Vec<WeightedSample>,
    sale_prices: Vec<WeightedSample>,
    buy_fill_hours: Vec<WeightedSample>,
    sell_fill_hours: Vec<WeightedSample>,
    open_age_hours: Vec<WeightedSample>,
    prediction_errors: Vec<WeightedSample>,
    prediction_abs_errors: Vec<WeightedSample>,
}

#[derive(Clone, Debug, Default)]
struct LearnedState {
    item_profiles: HashMap<String, ItemProfile>,
    category_profiles: HashMap<String, ItemProfile>,
    global: ItemProfile,
    transaction_count: usize,
}

#[derive(Clone, Debug)]
pub struct UltimateLearningEngine {
    config: UltimateConfig,
    learned: LearnedState,

    decisions: Vec<DecisionRecord>,
    snapshots: Vec<MarketSnapshot>,
    events: Vec<EventSignal>,
    graph_edges: Vec<GraphEdge>,
    relic_tables: Vec<RelicDropTable>,

    bandit: HashMap<(TradeSide, String, String), BanditEvidence>,
    embedding_history: Vec<(Vec<f64>, f64, f64)>,
    embedding_feature_weights: Vec<f64>,

    active_decision_ids: HashSet<String>,
    health: ModelHealth,

    pending_snapshots: Vec<MarketSnapshot>,
    pending_decisions: HashMap<String, DecisionRecord>,
    pending_outcomes: Vec<OutcomeRecord>,
    pending_alerts: Vec<AlertRecord>,
    config_dirty: bool,

    arbitrage_opportunities: Vec<ArbitrageOpportunity>,
    portfolio_state: PortfolioState,
}

impl UltimateLearningEngine {
    pub async fn load(
        db: &DatabaseConnection,
        my_orders: &OrderList<Order>,
    ) -> Result<Self, Error> {
        let config = LearningStore::load_config(db).await?;
        let decisions = LearningStore::load_decisions(db, MAX_DECISIONS_IN_MEMORY).await?;
        let snapshots = LearningStore::load_recent_snapshots(db, MAX_SNAPSHOTS_IN_MEMORY).await?;
        let events = if config.event_intelligence {
            match refresh_event_intelligence(db).await {
                Ok(events) => events,
                Err(_) => LearningStore::load_active_events(db).await?,
            }
        } else {
            LearningStore::load_active_events(db).await?
        };
        let graph_edges = LearningStore::load_graph_edges(db).await?;
        let relic_tables = if config.arbitrage_engine {
            relic_drop_tables().await.unwrap_or_default()
        } else {
            vec![]
        };

        let transactions = TransactionQuery::get_all(db, TransactionPaginationQueryDto::new(1, -1))
            .await?
            .results;

        let active_decision_ids: HashSet<String> = my_orders
            .buy_orders
            .iter()
            .chain(my_orders.sell_orders.iter())
            .map(|order| {
                order
                    .properties
                    .get_property_value("algoframe_decision_id", String::new())
            })
            .filter(|id| !id.is_empty())
            .collect();

        let learned = build_learned_state(&transactions, &config);
        let mut engine = Self {
            config,
            learned,
            decisions,
            snapshots,
            events,
            graph_edges,
            relic_tables,
            bandit: HashMap::new(),
            embedding_history: vec![],
            embedding_feature_weights: vec![],
            active_decision_ids,
            health: ModelHealth::default(),
            pending_snapshots: vec![],
            pending_decisions: HashMap::new(),
            pending_outcomes: vec![],
            pending_alerts: vec![],
            config_dirty: false,
            arbitrage_opportunities: vec![],
            portfolio_state: PortfolioState::default(),
        };

        engine.reconcile_outcomes(&transactions);
        engine.rebuild_learning_indexes();
        engine.embedding_feature_weights = learned_embedding_weights(&engine.decisions);
        engine.health = model_health(&engine.decisions, engine.config.max_drawdown_pct);
        engine.arbitrage_opportunities = engine.discover_arbitrage();

        if engine.config.automatic_rollback
            && engine.health.score < engine.config.minimum_model_health
        {
            engine.health.fallback_active = true;
            engine.queue_alert(
                "critical",
                "model_fallback",
                format!(
                    "Model health {:.0}% is below threshold {:.0}%; conservative fallback is active.",
                    engine.health.score * 100.0,
                    engine.config.minimum_model_health * 100.0
                ),
                None,
            );
        }

        Ok(engine)
    }

    pub fn config(&self) -> &UltimateConfig {
        &self.config
    }

    pub fn health(&self) -> &ModelHealth {
        &self.health
    }

    pub fn is_paper_mode(&self) -> bool {
        self.config.mode == OperatingMode::Paper
    }

    pub fn transaction_count(&self) -> usize {
        self.learned.transaction_count
    }

    pub fn decision_count(&self) -> usize {
        self.decisions.len() + self.pending_decisions.len()
    }

    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len() + self.pending_snapshots.len()
    }

    pub fn update_config(&mut self, config: UltimateConfig) {
        self.config = config.normalized();
        self.config.settings_revision = self.config.settings_revision.saturating_add(1);
        self.config_dirty = true;
    }

    pub fn observe_market(
        &mut self,
        item: &CacheTradableItem,
        sub_type: &Option<SubType>,
        price: &ItemPriceInfo,
        buy_prices: &[i64],
        sell_prices: &[i64],
    ) -> MarketSnapshot {
        let now = Utc::now();
        let key = item_key(&item.wfm_id, sub_type);
        let category = category_from(&item.tags, &item.name);

        let recent_item_snapshots: Vec<&MarketSnapshot> = self
            .snapshots
            .iter()
            .chain(self.pending_snapshots.iter())
            .rev()
            .filter(|snapshot| snapshot.item_key == key)
            .take(32)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let recent_mid_prices: Vec<f64> = recent_item_snapshots
            .iter()
            .map(|snapshot| snapshot.features.mid_price)
            .filter(|value| *value > 0.0)
            .collect();

        let recent_dynamics_input: Vec<(DateTime<Utc>, i64, i64, i64, i64)> = recent_item_snapshots
            .iter()
            .map(|snapshot| {
                (
                    snapshot.created_at,
                    snapshot.features.robust_bid,
                    snapshot.features.robust_ask,
                    snapshot.features.buy_depth,
                    snapshot.features.sell_depth,
                )
            })
            .collect();

        let recent_forecast_input: Vec<(DateTime<Utc>, f64, f64)> = recent_item_snapshots
            .iter()
            .map(|snapshot| {
                (
                    snapshot.created_at,
                    snapshot.features.mid_price,
                    snapshot.features.spread,
                )
            })
            .collect();

        let dynamics = order_book_dynamics(&recent_dynamics_input);
        let forecast = forecast_prices(&recent_forecast_input);

        let anomaly = if self.config.anomaly_detection {
            anomaly_report(
                buy_prices,
                sell_prices,
                &recent_mid_prices,
                dynamics.churn_score,
            )
        } else {
            AnomalyReport::default()
        };

        let historical = historical_center(price);
        let quality = data_quality(buy_prices, sell_prices, historical, &anomaly, 0.0);

        let (robust_bid, _) = robust_ceiling_price(buy_prices);
        let (robust_ask, _) = robust_floor_price(sell_prices);

        let best_bid = buy_prices.iter().copied().max().unwrap_or(0);
        let best_ask = sell_prices.iter().copied().min().unwrap_or(0);

        let mid_price = match (robust_bid > 0, robust_ask > 0) {
            (true, true) => (robust_bid as f64 + robust_ask as f64) / 2.0,
            (true, false) => robust_bid as f64,
            (false, true) => robust_ask as f64,
            _ => historical,
        };

        let spread = if robust_bid > 0 && robust_ask > 0 {
            (robust_ask - robust_bid) as f64
        } else {
            0.0
        };

        let spread_percent = if robust_ask > 0 {
            (spread / robust_ask as f64).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let volatility = price_volatility(price);
        let (regime, regime_score) = if self.config.regime_detection {
            detect_regime(
                mid_price,
                historical,
                price.week_price_shift,
                &recent_mid_prices,
            )
        } else {
            (MarketRegime::Normal, 0.0)
        };

        let profile = self.contextual_profile(&key, &category);
        let inventory_pressure = inventory_pressure(&profile);

        let event_signal = if self.config.event_intelligence {
            self.event_signal(item)
        } else {
            0.0
        };

        let arbitrage_signal = if self.config.arbitrage_engine {
            self.arbitrage_signal(&key, &item.name)
        } else {
            0.0
        };

        let opportunity_cost_signal = if self.config.opportunity_forecasting {
            self.market_wide_opportunity_cost()
        } else {
            0.0
        };

        let lifecycle = self.item_lifecycle(item, &profile);

        let (time_hour_sin, time_hour_cos, weekday_sin, weekday_cos) = time_features(now);

        let mut features = MarketFeatures {
            best_bid,
            best_ask,
            robust_bid,
            robust_ask,
            mid_price,
            spread,
            spread_percent,
            buy_depth: buy_prices.len() as i64,
            sell_depth: sell_prices.len() as i64,
            volume: price.volume,
            liquidity: liquidity_score(price.volume),
            volatility,
            week_shift: price.week_price_shift,
            inventory_pressure,
            time_hour_sin,
            time_hour_cos,
            weekday_sin,
            weekday_cos,
            event_signal,
            arbitrage_signal,
            opportunity_cost_signal,
            regime,
            regime_score,
            lifecycle,
            anomaly,
            quality,
            dynamics,
            forecast,
            embedding: vec![],
        };

        features.embedding = market_embedding(&features);

        let mut snapshot = MarketSnapshot {
            id: Uuid::new_v4().to_string(),
            item_key: key,
            wfm_id: item.wfm_id.clone(),
            wfm_url: item.wfm_url.clone(),
            item_name: item.name.clone(),
            category,
            tags: item.tags.clone(),
            sub_type: serde_json::to_value(sub_type).unwrap_or(serde_json::Value::Null),
            created_at: now,
            external_only: true,
            buy_prices: buy_prices.to_vec(),
            sell_prices: sell_prices.to_vec(),
            features,
            model_version: MODEL_VERSION.to_string(),
            feature_schema_version: FEATURE_SCHEMA_VERSION,
        };

        if self.config.book_persistence_model {
            let mut history = self.snapshots.clone();
            history.extend(self.pending_snapshots.iter().cloned());
            history.push(snapshot.clone());

            let curve = liquidity_curve(&history, &snapshot.item_key);
            snapshot.features.liquidity = (snapshot.features.liquidity * 0.70
                + curve.real_liquidity_score * 0.30)
                .clamp(0.0, 1.0);

            snapshot.features.quality.score *=
                (1.0 - curve.replacement_frequency * 0.20).clamp(0.60, 1.0);
        }

        if self.config.learned_market_states {
            let mut history = self.snapshots.clone();
            history.extend(self.pending_snapshots.iter().cloned());

            let latent = learned_market_state(&snapshot, &history);

            if latent.label.contains("post-shock") || latent.label.contains("seller collapse") {
                snapshot.features.regime = MarketRegime::Shock;
                snapshot.features.regime_score =
                    snapshot.features.regime_score.max(latent.confidence);
            } else if latent.label.contains("bidding-war")
                || latent.label.contains("rapid appreciation")
            {
                if snapshot.features.regime == MarketRegime::Normal {
                    snapshot.features.regime = MarketRegime::Moving;
                }

                snapshot.features.regime_score =
                    snapshot.features.regime_score.max(latent.confidence * 0.75);
            }
        }

        snapshot.features.embedding = market_embedding(&snapshot.features);

        if self.config.market_recording {
            self.pending_snapshots.push(snapshot.clone());
        }

        snapshot
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan_buy(
        &mut self,
        item: &CacheTradableItem,
        sub_type: &Option<SubType>,
        price: &ItemPriceInfo,
        snapshot: &MarketSnapshot,
        minimum_profit: i64,
        minimum_margin_percent: i64,
        requested_quantity: i64,
        total_capital_cap: i64,
    ) -> ExecutionIntent {
        self.plan_trade(
            TradeSide::Buy,
            item,
            sub_type,
            price,
            snapshot,
            minimum_profit,
            minimum_margin_percent,
            requested_quantity,
            total_capital_cap,
            0,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan_sell(
        &mut self,
        item: &CacheTradableItem,
        sub_type: &Option<SubType>,
        price: &ItemPriceInfo,
        snapshot: &MarketSnapshot,
        minimum_profit: i64,
        requested_quantity: i64,
        bought_price: i64,
        current_order_price: i64,
    ) -> ExecutionIntent {
        self.plan_trade(
            TradeSide::Sell,
            item,
            sub_type,
            price,
            snapshot,
            minimum_profit,
            -1,
            requested_quantity,
            i64::MAX,
            bought_price,
            current_order_price,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_trade(
        &mut self,
        side: TradeSide,
        item: &CacheTradableItem,
        sub_type: &Option<SubType>,
        price: &ItemPriceInfo,
        snapshot: &MarketSnapshot,
        minimum_profit: i64,
        minimum_margin_percent: i64,
        requested_quantity: i64,
        total_capital_cap: i64,
        bought_price: i64,
        current_order_price: i64,
    ) -> ExecutionIntent {
        let decision_id = Uuid::new_v4().to_string();
        let item_key = snapshot.item_key.clone();
        let category = snapshot.category.clone();

        let mut explanation = Explanation {
            headline: format!(
                "{} evaluation for {}",
                match side {
                    TradeSide::Buy => "WTB",
                    TradeSide::Sell => "WTS",
                },
                item.name
            ),
            ..Default::default()
        };

        if !self.config.enabled {
            explanation
                .negative_factors
                .push("AlgoFrame learning is disabled".into());
            return ExecutionIntent::rejected(
                decision_id,
                snapshot.id.clone(),
                side,
                item.wfm_id.clone(),
                item_key,
                item.name.clone(),
                category,
                explanation,
            );
        }

        if snapshot.features.quality.score < self.config.minimum_data_quality {
            explanation.negative_factors.push(format!(
                "data quality {:.0}% is below minimum {:.0}%",
                snapshot.features.quality.score * 100.0,
                self.config.minimum_data_quality * 100.0,
            ));
            explanation
                .negative_factors
                .extend(snapshot.features.quality.reasons.clone());

            let intent = ExecutionIntent::rejected(
                decision_id,
                snapshot.id.clone(),
                side,
                item.wfm_id.clone(),
                item_key,
                item.name.clone(),
                category,
                explanation,
            );
            self.record_rejected_intent(&intent, snapshot);
            return intent;
        }

        let fallback_active = self.health.fallback_active;
        let profile = self.contextual_profile(&item_key, &category);
        let effective_confidence =
            profile.confidence * snapshot.features.regime.confidence_multiplier();

        let seed = make_seed(&[
            &decision_id,
            &snapshot.id,
            &self.config.settings_revision.to_string(),
        ]);

        let model_role = if fallback_active {
            ModelRole::Fallback
        } else if stable_fraction(&[&item_key, &snapshot.id]) < self.config.challenger_fraction {
            ModelRole::Challenger
        } else {
            ModelRole::Champion
        };

        let policy_name = match model_role {
            ModelRole::Fallback => "conservative".to_string(),
            ModelRole::Champion => self.config.champion_policy.clone(),
            ModelRole::Challenger => self.config.challenger_policy.clone(),
            ModelRole::Shadow => "balanced".to_string(),
        };

        let family = family_from_name(&item.name);
        let context_key = context_key(side, &category, &family, &item_key, &snapshot.features);

        let minimum_profit = if minimum_profit <= -1 {
            0
        } else {
            minimum_profit.max(0)
        };

        let max_quantity = requested_quantity
            .max(1)
            .min(self.config.max_trade_quantity)
            .min(self.config.max_quantity_search);

        let mut candidates = match side {
            TradeSide::Buy => self.generate_buy_candidates(
                &context_key,
                snapshot,
                price,
                &profile,
                minimum_profit,
                minimum_margin_percent,
                max_quantity,
            ),
            TradeSide::Sell => self.generate_sell_candidates(
                &context_key,
                snapshot,
                price,
                &profile,
                minimum_profit,
                max_quantity,
                bought_price,
                current_order_price,
            ),
        };

        let capital_budget = if total_capital_cap <= 0 {
            1_000_000_000.0
        } else {
            total_capital_cap as f64
        };
        for candidate in &mut candidates {
            self.apply_candidate_guardrails(
                candidate,
                snapshot,
                capital_budget,
                &profile,
                fallback_active,
            );
        }

        candidates.retain(|candidate| candidate.valid);

        if candidates.is_empty() {
            explanation
                .negative_factors
                .push("no price/quantity combination passed the execution guardrails".into());

            let intent = ExecutionIntent::rejected(
                decision_id,
                snapshot.id.clone(),
                side,
                item.wfm_id.clone(),
                item_key,
                item.name.clone(),
                category,
                explanation,
            );
            self.record_rejected_intent(&intent, snapshot);
            return intent;
        }

        let candidate_keys: Vec<String> = candidates
            .iter()
            .map(|candidate| candidate.key.clone())
            .collect();

        let evidence_map: HashMap<String, BanditEvidence> = candidate_keys
            .iter()
            .map(|key| {
                (
                    key.clone(),
                    self.hierarchical_evidence(side, &context_key, &category, key),
                )
            })
            .collect();

        let candidate_map: HashMap<String, CandidateAction> = candidates
            .iter()
            .map(|candidate| (candidate.key.clone(), candidate.clone()))
            .collect();

        let propensities = if fallback_active {
            let conservative = conservative_candidate(side, &candidates);
            let mut result = HashMap::new();
            for candidate in &candidates {
                result.insert(
                    candidate.key.clone(),
                    if candidate.key == conservative.key {
                        1.0
                    } else {
                        0.0
                    },
                );
            }
            result
        } else {
            estimate_propensities(
                &candidate_keys,
                self.config.propensity_samples,
                seed,
                |key, rng| {
                    let candidate = candidate_map.get(key).unwrap();
                    let evidence = evidence_map.get(key).unwrap();

                    let base = thompson_utility(
                        evidence,
                        action_prior(side, candidate.price_offset),
                        candidate.reward.final_reward,
                        candidate.uncertainty,
                        rng,
                    );

                    let nearest = nearest_neighbor_reward(
                        &snapshot.features.embedding,
                        &self.embedding_history,
                        self.config.nearest_neighbor_count,
                        &self.embedding_feature_weights,
                    );

                    let blended = base * (1.0 - nearest.1 * 0.25) + nearest.0 * nearest.1 * 0.25;

                    apply_policy_utility(&policy_name, side, candidate, blended)
                },
            )
        };

        let mut rng = StdRng::seed_from_u64(seed);

        let selected = if fallback_active {
            conservative_candidate(side, &candidates)
        } else {
            candidates
                .iter()
                .max_by(|a, b| {
                    let evidence_a = evidence_map.get(&a.key).unwrap();
                    let evidence_b = evidence_map.get(&b.key).unwrap();

                    let ua = apply_policy_utility(
                        &policy_name,
                        side,
                        a,
                        thompson_utility(
                            evidence_a,
                            action_prior(side, a.price_offset),
                            a.reward.final_reward,
                            a.uncertainty,
                            &mut rng,
                        ),
                    );
                    let ub = apply_policy_utility(
                        &policy_name,
                        side,
                        b,
                        thompson_utility(
                            evidence_b,
                            action_prior(side, b.price_offset),
                            b.reward.final_reward,
                            b.uncertainty,
                            &mut rng,
                        ),
                    );

                    ua.partial_cmp(&ub).unwrap_or(Ordering::Equal)
                })
                .unwrap()
                .clone()
        };

        // Learned no-trade baseline. After enough counterfactual skip outcomes,
        // AlgoFrame can explicitly prefer preserving capital over a weak trade.
        let skip_evidence = self.hierarchical_evidence(side, &context_key, &category, "skip");

        if !fallback_active
            && side == TradeSide::Buy
            && skip_evidence.reward_samples.len() >= 8
            && skip_evidence.reward_mean() > selected.reward.final_reward
        {
            explanation.negative_factors.push(format!(
                "learned skip baseline {:.4} exceeds best trade reward {:.4}",
                skip_evidence.reward_mean(),
                selected.reward.final_reward,
            ));
            explanation
                .confidence_notes
                .push("decision learned from prior skipped-opportunity replays".into());

            let mut intent = ExecutionIntent::rejected(
                decision_id,
                snapshot.id.clone(),
                side,
                item.wfm_id.clone(),
                item_key,
                item.name.clone(),
                category,
                explanation,
            );

            intent.paper = self.config.mode == OperatingMode::Paper;
            intent.context_key = context_key;
            intent.selected_action = "skip".to_string();
            intent.selected_propensity = 1.0;
            intent.propensities = HashMap::from([("skip".to_string(), 1.0)]);
            intent.regime = snapshot.features.regime;
            intent.data_quality = snapshot.features.quality.score;
            intent.anomaly_score = snapshot.features.anomaly.score;
            intent.volatility = snapshot.features.volatility;
            intent.model_role = model_role;
            intent.policy_name = policy_name;
            intent.seed = seed;
            intent.settings_hash = settings_hash(&self.config);

            self.record_rejected_intent(&intent, snapshot);
            return intent;
        }

        let selected_propensity = propensities
            .get(&selected.key)
            .copied()
            .unwrap_or(0.01)
            .max(0.01);

        let shadow_actions = if self.config.shadow_mode {
            HashMap::from([
                (
                    "shadow_conservative".to_string(),
                    best_candidate_for_policy("conservative", side, &candidates)
                        .key
                        .clone(),
                ),
                (
                    "shadow_aggressive".to_string(),
                    best_candidate_for_policy("aggressive", side, &candidates)
                        .key
                        .clone(),
                ),
                (
                    "shadow_fast_turnover".to_string(),
                    best_candidate_for_policy("fast_turnover", side, &candidates)
                        .key
                        .clone(),
                ),
                (
                    "shadow_max_reward".to_string(),
                    best_candidate_for_policy("max_reward", side, &candidates)
                        .key
                        .clone(),
                ),
                (
                    "shadow_balanced".to_string(),
                    best_candidate_for_policy("balanced", side, &candidates)
                        .key
                        .clone(),
                ),
            ])
        } else {
            HashMap::new()
        };

        explanation.positive_factors.push(format!(
            "selected {} at {}p × {}",
            selected.key, selected.price, selected.quantity
        ));
        explanation.positive_factors.push(format!(
            "expected {:.1}p profit; {:.2}% expected return/hour",
            selected.reward.expected_profit, selected.reward.final_reward
        ));
        explanation.positive_factors.push(format!(
            "{:.0}% chance to fill within 1h",
            selected.survival.fill_1h * 100.0
        ));

        if self.config.distributional_predictions {
            let distribution = profit_distribution(&self.decisions, Some(&category), None);

            if distribution.sample_count >= 6 {
                explanation.confidence_notes.push(format!(
                    "profit P10/P50/P90 {:.1}/{:.1}/{:.1}p; loss risk {:.0}%",
                    distribution.p10,
                    distribution.p50,
                    distribution.p90,
                    distribution.loss_probability * 100.0
                ));

                if self.config.conformal_intervals {
                    explanation.confidence_notes.push(format!(
                        "80% conformal interval {:.1}..{:.1}p",
                        distribution.conformal_low_80, distribution.conformal_high_80
                    ));
                }

                if self.config.uncertainty_decomposition {
                    explanation.confidence_notes.push(format!(
                        "aleatoric {:.0}% / epistemic {:.0}% uncertainty",
                        distribution.aleatoric_uncertainty.clamp(0.0, 1.0) * 100.0,
                        distribution.epistemic_uncertainty * 100.0
                    ));
                }
            }
        }

        if snapshot.features.arbitrage_signal > 0.05 {
            explanation.positive_factors.push(format!(
                "related-item arbitrage signal +{:.0}%",
                snapshot.features.arbitrage_signal * 100.0
            ));
        }

        if snapshot.features.event_signal.abs() > 0.05 {
            explanation.positive_factors.push(format!(
                "event intelligence signal {:+.0}%",
                snapshot.features.event_signal * 100.0
            ));
        }

        if snapshot.features.anomaly.score > 0.25 {
            explanation.negative_factors.push(format!(
                "market anomaly risk {:.0}%",
                snapshot.features.anomaly.score * 100.0
            ));
        }

        if snapshot.features.inventory_pressure > 0.25 {
            explanation.negative_factors.push(format!(
                "inventory pressure {:.0}%",
                snapshot.features.inventory_pressure * 100.0
            ));
        }

        if fallback_active {
            explanation
                .guardrails
                .push("model health fallback forced conservative execution".into());
        }

        explanation.confidence_notes.push(format!(
            "item/context confidence {:.0}%; data quality {:.0}%",
            effective_confidence * 100.0,
            snapshot.features.quality.score * 100.0
        ));
        explanation.confidence_notes.push(format!(
            "market regime: {}",
            snapshot.features.regime.as_str()
        ));

        let intent = ExecutionIntent {
            allowed: true,
            paper: self.config.mode == OperatingMode::Paper,
            decision_id: decision_id.clone(),
            snapshot_id: snapshot.id.clone(),
            side,
            lifecycle: LifecycleState::Allocated,
            wfm_id: item.wfm_id.clone(),
            item_key: item_key.clone(),
            item_name: item.name.clone(),
            category: category.clone(),
            price: selected.price,
            quantity: selected.quantity,
            capital: match side {
                TradeSide::Buy => selected.price.max(1) as f64 * selected.quantity as f64,
                TradeSide::Sell => bought_price.max(1) as f64 * selected.quantity as f64,
            },
            expected_sell_price: selected.expected_sell_price,
            survival: selected.survival.clone(),
            reward: selected.reward.clone(),
            confidence: effective_confidence,
            uncertainty: selected.uncertainty,
            context_key: context_key.clone(),
            selected_action: selected.key.clone(),
            selected_propensity,
            propensities: propensities.clone(),
            shadow_actions: shadow_actions.clone(),
            regime: snapshot.features.regime,
            data_quality: snapshot.features.quality.score,
            anomaly_score: snapshot.features.anomaly.score,
            volatility: snapshot.features.volatility,
            model_role,
            policy_name: policy_name.clone(),
            explanation: explanation.clone(),
            seed,
            model_version: MODEL_VERSION.to_string(),
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            reward_version: REWARD_VERSION,
            policy_version: POLICY_VERSION,
            settings_hash: settings_hash(&self.config),
        };

        let now = Utc::now();
        let decision = DecisionRecord {
            id: decision_id,
            snapshot_id: snapshot.id.clone(),
            wfm_id: item.wfm_id.clone(),
            item_key,
            item_name: item.name.clone(),
            category,
            side,
            lifecycle: if intent.paper {
                LifecycleState::PaperSimulated
            } else {
                LifecycleState::Allocated
            },
            status: if intent.paper {
                DecisionStatus::PaperOpen
            } else {
                DecisionStatus::Open
            },
            chosen_action: selected.key,
            price: selected.price,
            quantity: selected.quantity,
            filled_quantity: 0,
            capital: intent.capital,
            context_key,
            propensities,
            chosen_propensity: selected_propensity,
            shadow_actions,
            predicted_profit: selected.reward.expected_profit,
            predicted_reward: selected.reward.final_reward,
            predicted_fill: selected.survival,
            actual_profit: None,
            actual_reward: None,
            actual_fill_hours: None,
            actual_cycle_hours: None,
            fill_ratio: 0.0,
            features: snapshot.features.clone(),
            reward_breakdown: selected.reward,
            explanation,
            regime: snapshot.features.regime,
            model_role,
            policy_name,
            seed,
            model_version: MODEL_VERSION.to_string(),
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            reward_version: REWARD_VERSION,
            policy_version: POLICY_VERSION,
            settings_hash: settings_hash(&self.config),
            created_at: now,
            updated_at: now,
            filled_at: None,
            completed_at: None,
        };

        self.pending_decisions
            .insert(decision.id.clone(), decision.clone());

        self.decisions.push(decision);

        intent
    }

    pub fn apply_intent_to_properties(
        &self,
        properties: &mut WfmProperties,
        intent: &ExecutionIntent,
    ) {
        properties.set_property_value("algoframe_model", &intent.model_version);
        properties.set_property_value(
            "algoframe_feature_schema_version",
            intent.feature_schema_version,
        );
        properties.set_property_value("algoframe_reward_version", intent.reward_version);
        properties.set_property_value("algoframe_policy_version", intent.policy_version);
        properties.set_property_value("algoframe_settings_hash", &intent.settings_hash);

        properties.set_property_value("algoframe_decision_id", &intent.decision_id);
        properties.set_property_value("algoframe_snapshot_id", &intent.snapshot_id);
        properties.set_property_value("algoframe_context_key", &intent.context_key);
        properties.set_property_value("algoframe_action", &intent.selected_action);
        properties.set_property_value("algoframe_propensity", intent.selected_propensity);
        properties.set_property_value("algoframe_propensities", &intent.propensities);
        properties.set_property_value("algoframe_shadow_actions", &intent.shadow_actions);

        properties.set_property_value("algoframe_expected_sell", intent.expected_sell_price);
        properties.set_property_value("algoframe_expected_profit", intent.reward.expected_profit);
        properties.set_property_value("algoframe_predicted_reward", intent.reward.final_reward);
        properties.set_property_value("algoframe_predicted_fill", &intent.survival);

        properties.set_property_value("algoframe_learning_confidence", intent.confidence);
        properties.set_property_value("algoframe_uncertainty", intent.uncertainty);
        properties.set_property_value("algoframe_data_quality", intent.data_quality);
        properties.set_property_value("algoframe_anomaly_score", intent.anomaly_score);
        properties.set_property_value("algoframe_regime", intent.regime.as_str());
        properties.set_property_value("algoframe_policy_name", &intent.policy_name);
        properties.set_property_value("algoframe_category", &intent.category);
        properties.set_property_value("algoframe_family", family_from_name(&intent.item_name));
        properties.set_property_value("algoframe_volatility", intent.volatility);
        properties.set_property_value("algoframe_explanation", &intent.explanation);
        properties.set_property_value("algoframe_seed", intent.seed);

        properties.set_property_value(
            "allocation_value",
            intent.reward.final_reward.max(0.0) * intent.capital.max(1.0),
        );

        let existing_started_at: String =
            properties.get_property_value("algoframe_order_started_at", String::new());
        if existing_started_at.is_empty() {
            properties.set_property_value("algoframe_order_started_at", Utc::now().to_rfc3339());
        }
    }

    pub fn apply_repricing_hysteresis(
        &mut self,
        intent: &mut ExecutionIntent,
        current_order_price: i64,
        properties: &WfmProperties,
        snapshot: &MarketSnapshot,
        bought_price: i64,
    ) {
        if !intent.allowed || current_order_price <= 0 || current_order_price == intent.price {
            return;
        }

        let age_minutes = Self::reprice_age_hours(properties) * 60.0;

        let required_minutes = if snapshot.features.regime == MarketRegime::Shock {
            self.config.min_reprice_minutes * 0.50
        } else if snapshot.features.dynamics.churn_score > 0.70
            || snapshot.features.dynamics.competitive_response_score > 0.70
        {
            self.config.stable_reprice_minutes * 1.50
        } else if snapshot.features.liquidity < 0.35 {
            self.config.stable_reprice_minutes
        } else {
            self.config.min_reprice_minutes
        }
        .clamp(0.0, 240.0);

        if age_minutes > 0.0 && age_minutes < required_minutes {
            self.revise_intent_price(
                intent,
                current_order_price,
                bought_price,
                format!(
                    "adaptive repricing cooldown ({:.1}/{:.1} min)",
                    age_minutes, required_minutes
                ),
            );
            return;
        }

        let quantity = intent.quantity.max(1);
        let current_expected_profit = match intent.side {
            TradeSide::Buy => intent.expected_sell_price - current_order_price as f64,
            TradeSide::Sell => (current_order_price - bought_price).max(0) as f64,
        };

        let distance = (intent.price - current_order_price).abs() as f64;

        let current_fill_hours = match intent.side {
            TradeSide::Buy => {
                if intent.price > current_order_price {
                    // A more competitive WTB should fill faster.
                    intent.survival.expected_fill_hours * (0.90_f64).powf(distance)
                } else {
                    // A lower WTB sacrifices queue/price priority.
                    intent.survival.expected_fill_hours * (1.0 + 0.20 * distance)
                }
            }
            TradeSide::Sell => {
                if intent.price < current_order_price {
                    intent.survival.expected_fill_hours * (0.90_f64).powf(distance)
                } else {
                    intent.survival.expected_fill_hours * (1.0 + 0.20 * distance)
                }
            }
        }
        .clamp(0.05, 720.0);

        let current_capital = match intent.side {
            TradeSide::Buy => current_order_price.max(1) as f64 * quantity as f64,
            TradeSide::Sell => bought_price.max(1) as f64 * quantity as f64,
        };

        let current_reward = normalized_reward(
            current_expected_profit * quantity as f64,
            current_capital.max(1.0),
            current_fill_hours,
            self.config.human_minutes_per_trade,
            self.config.human_time_value_plat_per_hour,
        );

        let minimum_gain = current_reward.abs() * self.config.reprice_min_reward_gain_pct + 0.005;

        if intent.reward.final_reward <= current_reward + minimum_gain {
            self.revise_intent_price(
                intent,
                current_order_price,
                bought_price,
                format!(
                    "repricing hysteresis: reward gain {:.4} below {:.4}",
                    intent.reward.final_reward - current_reward,
                    minimum_gain
                ),
            );
        }
    }

    pub fn reprice_age_hours(properties: &WfmProperties) -> f64 {
        let last_repriced: String =
            properties.get_property_value("algoframe_last_repriced_at", String::new());

        let started: String =
            properties.get_property_value("algoframe_order_started_at", String::new());

        let timestamp = if !last_repriced.is_empty() {
            last_repriced
        } else {
            started
        };

        if timestamp.is_empty() {
            return 0.0;
        }

        let Ok(parsed) = DateTime::parse_from_rfc3339(&timestamp) else {
            return 0.0;
        };

        (Utc::now() - parsed.with_timezone(&Utc))
            .num_seconds()
            .max(0) as f64
            / 3600.0
    }

    pub fn mark_repriced(
        properties: &mut WfmProperties,
        current_order_price: i64,
        next_order_price: i64,
    ) {
        if current_order_price <= 0 || current_order_price != next_order_price {
            properties.set_property_value("algoframe_last_repriced_at", Utc::now().to_rfc3339());
        }
    }

    pub fn reject_intent(&mut self, intent: &mut ExecutionIntent, reason: impl Into<String>) {
        let reason = reason.into();
        intent.allowed = false;
        intent.lifecycle = LifecycleState::Failed;
        intent.explanation.guardrails.push(reason.clone());

        if let Some(decision) = self
            .decisions
            .iter_mut()
            .find(|decision| decision.id == intent.decision_id)
        {
            decision.status = DecisionStatus::Rejected;
            decision.lifecycle = LifecycleState::Failed;
            decision.explanation.guardrails.push(reason);
            decision.updated_at = Utc::now();

            self.pending_decisions
                .insert(decision.id.clone(), decision.clone());
        }
    }

    pub fn record_guardrail_skip(
        &mut self,
        side: TradeSide,
        item: &CacheTradableItem,
        sub_type: &Option<SubType>,
        snapshot: &MarketSnapshot,
        reason: impl Into<String>,
    ) {
        let reason = reason.into();
        let explanation = Explanation {
            headline: format!(
                "{} skipped by execution guardrail",
                match side {
                    TradeSide::Buy => "WTB",
                    TradeSide::Sell => "WTS",
                }
            ),
            negative_factors: vec![reason],
            ..Default::default()
        };

        let intent = ExecutionIntent::rejected(
            Uuid::new_v4().to_string(),
            snapshot.id.clone(),
            side,
            item.wfm_id.clone(),
            item_key(&item.wfm_id, sub_type),
            item.name.clone(),
            snapshot.category.clone(),
            explanation,
        );

        self.record_rejected_intent(&intent, snapshot);
    }

    pub fn revise_intent_price(
        &mut self,
        intent: &mut ExecutionIntent,
        new_price: i64,
        bought_price: i64,
        reason: impl Into<String>,
    ) {
        if !intent.allowed || new_price <= 0 || new_price == intent.price {
            return;
        }

        let old_price = intent.price;
        intent.price = new_price;

        // A hard execution guardrail changed the action that will actually hit
        // the market. Re-label it so outcomes are never learned against the
        // pre-guardrail action.
        let revised_action = format!(
            "{}@{}:q{}",
            match intent.side {
                TradeSide::Buy => "buy",
                TradeSide::Sell => "sell",
            },
            new_price,
            intent.quantity.max(1),
        );
        intent.selected_action = revised_action.clone();
        intent.selected_propensity = 1.0;
        intent.propensities.clear();
        intent.propensities.insert(revised_action.clone(), 1.0);

        let expected_profit = match intent.side {
            TradeSide::Buy => intent.expected_sell_price - new_price as f64,
            TradeSide::Sell => (new_price - bought_price).max(0) as f64,
        };

        intent.reward.expected_profit = expected_profit;
        intent.reward.expected_roi = if intent.side == TradeSide::Buy {
            expected_profit / new_price.max(1) as f64
        } else {
            expected_profit / bought_price.max(1) as f64
        };

        intent.capital = match intent.side {
            TradeSide::Buy => new_price.max(1) as f64 * intent.quantity.max(1) as f64,
            TradeSide::Sell => bought_price.max(1) as f64 * intent.quantity.max(1) as f64,
        };

        let cycle_hours = intent.survival.expected_fill_hours.max(0.05);
        intent.reward.final_reward = normalized_reward(
            expected_profit * intent.quantity.max(1) as f64,
            intent.capital.max(1.0),
            cycle_hours,
            self.config.human_minutes_per_trade,
            self.config.human_time_value_plat_per_hour,
        );

        intent.explanation.guardrails.push(format!(
            "{} adjusted price {}p → {}p",
            reason.into(),
            old_price,
            new_price
        ));

        if let Some(decision) = self
            .decisions
            .iter_mut()
            .find(|decision| decision.id == intent.decision_id)
        {
            decision.price = new_price;
            decision.chosen_action = revised_action;
            decision.chosen_propensity = 1.0;
            decision.propensities = intent.propensities.clone();
            decision.capital = intent.capital;
            decision.predicted_profit = expected_profit;
            decision.predicted_reward = intent.reward.final_reward;
            decision.reward_breakdown = intent.reward.clone();
            decision.explanation = intent.explanation.clone();
            decision.updated_at = Utc::now();

            self.pending_decisions
                .insert(decision.id.clone(), decision.clone());
        }
    }

    fn record_rejected_intent(&mut self, intent: &ExecutionIntent, snapshot: &MarketSnapshot) {
        let now = Utc::now();

        let decision = DecisionRecord {
            id: intent.decision_id.clone(),
            snapshot_id: snapshot.id.clone(),
            wfm_id: intent.wfm_id.clone(),
            item_key: intent.item_key.clone(),
            item_name: intent.item_name.clone(),
            category: intent.category.clone(),
            side: intent.side,
            lifecycle: LifecycleState::Failed,
            status: DecisionStatus::Rejected,
            chosen_action: "skip".into(),
            price: 0,
            quantity: 0,
            filled_quantity: 0,
            capital: 0.0,
            context_key: if intent.context_key.is_empty() {
                context_key(
                    intent.side,
                    &snapshot.category,
                    &family_from_name(&snapshot.item_name),
                    &snapshot.item_key,
                    &snapshot.features,
                )
            } else {
                intent.context_key.clone()
            },
            propensities: HashMap::from([("skip".to_string(), 1.0)]),
            chosen_propensity: 1.0,
            shadow_actions: intent.shadow_actions.clone(),
            predicted_profit: 0.0,
            predicted_reward: 0.0,
            predicted_fill: SurvivalPrediction::default(),
            actual_profit: None,
            actual_reward: None,
            actual_fill_hours: None,
            actual_cycle_hours: None,
            fill_ratio: 0.0,
            features: snapshot.features.clone(),
            reward_breakdown: RewardBreakdown::default(),
            explanation: intent.explanation.clone(),
            regime: snapshot.features.regime,
            model_role: intent.model_role,
            policy_name: intent.policy_name.clone(),
            seed: intent.seed,
            model_version: MODEL_VERSION.to_string(),
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            reward_version: REWARD_VERSION,
            policy_version: POLICY_VERSION,
            settings_hash: settings_hash(&self.config),
            created_at: now,
            updated_at: now,
            filled_at: None,
            completed_at: None,
        };

        self.pending_decisions
            .insert(decision.id.clone(), decision.clone());
        self.decisions.push(decision);
    }

    pub fn order_age_hours(properties: &WfmProperties) -> f64 {
        let started: String =
            properties.get_property_value("algoframe_order_started_at", String::new());

        if started.is_empty() {
            return 0.0;
        }

        let Ok(started) = DateTime::parse_from_rfc3339(&started) else {
            return 0.0;
        };

        (Utc::now() - started.with_timezone(&Utc))
            .num_seconds()
            .max(0) as f64
            / 3600.0
    }

    pub fn portfolio_select(
        &mut self,
        orders: &OrderList<Order>,
        max_total_capital: i64,
        excluded_wfm_ids: &HashSet<String>,
    ) -> PortfolioDecision {
        let base_cap = max_total_capital.max(1) as f64;

        let positive_forecast = self
            .snapshots
            .iter()
            .rev()
            .take(100)
            .map(|snapshot| snapshot.features.forecast.opportunity_2h.max(0.0))
            .sum::<f64>()
            / self.snapshots.len().min(100).max(1) as f64;

        let dynamic_reserve = (self.config.cash_reserve_pct + positive_forecast * 0.15)
            .clamp(self.config.cash_reserve_pct, 0.60);

        let usable_capital = base_cap * (1.0 - dynamic_reserve);

        let mut candidates: Vec<(&Order, f64, f64, String, String, f64, f64)> = orders
            .buy_orders
            .iter()
            .filter(|order| !excluded_wfm_ids.contains(&order.item_id))
            .map(|order| {
                let capital = order.platinum as f64 * (order.quantity as f64).max(1.0);

                let value: f64 = order.properties.get_property_value("allocation_value", 0.0);

                let category: String = order
                    .properties
                    .get_property_value("algoframe_category", "other".to_string());

                let family: String = order
                    .properties
                    .get_property_value("algoframe_family", "other".to_string());

                let volatility: f64 = order
                    .properties
                    .get_property_value("algoframe_volatility", 0.0);

                let uncertainty: f64 = order
                    .properties
                    .get_property_value("algoframe_uncertainty", 1.0);

                let efficiency = if capital > 0.0 { value / capital } else { 0.0 };

                (
                    order,
                    capital,
                    efficiency,
                    category,
                    family,
                    volatility,
                    uncertainty,
                )
            })
            .collect();

        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

        let mut selected = Vec::new();
        let mut rejected = Vec::new();
        let mut reasons = HashMap::new();

        let mut state = PortfolioState {
            total_capital_budget: base_cap,
            reserved_capital: base_cap - usable_capital,
            ..Default::default()
        };

        for (order, capital, _, category, family, volatility, uncertainty) in candidates {
            let item_cap = *state.item_exposure.get(&order.item_id).unwrap_or(&0.0);
            let family_cap = *state.family_exposure.get(&family).unwrap_or(&0.0);
            let category_cap = *state.category_exposure.get(&category).unwrap_or(&0.0);

            let experimental = uncertainty >= 0.55;
            let high_vol = volatility >= 0.50;

            let violates_total = state.allocated_capital + capital > usable_capital;
            let violates_item = item_cap + capital > base_cap * self.config.max_item_exposure_pct;
            let violates_family =
                family_cap + capital > base_cap * self.config.max_family_exposure_pct;
            let violates_category =
                category_cap + capital > base_cap * self.config.max_category_exposure_pct;
            let violates_high_vol = high_vol
                && state.high_volatility_capital + capital
                    > base_cap * self.config.max_high_volatility_exposure_pct;
            let violates_experimental = experimental
                && state.experimental_capital + capital
                    > base_cap * self.config.experimental_budget_pct;

            let reason = if violates_total {
                Some("cash reserve / total capital budget")
            } else if violates_item {
                Some("per-item exposure limit")
            } else if violates_family {
                Some("correlated item-family exposure limit")
            } else if violates_category {
                Some("category exposure limit")
            } else if violates_high_vol {
                Some("high-volatility risk budget")
            } else if violates_experimental {
                Some("experimental ML risk budget")
            } else {
                None
            };

            if let Some(reason) = reason {
                rejected.push(order.id.clone());
                reasons.insert(order.id.clone(), reason.to_string());
                continue;
            }

            selected.push(order.id.clone());
            state.allocated_capital += capital;
            *state
                .item_exposure
                .entry(order.item_id.clone())
                .or_default() += capital;
            *state.family_exposure.entry(family).or_default() += capital;
            *state.category_exposure.entry(category).or_default() += capital;

            if high_vol {
                state.high_volatility_capital += capital;
            }
            if experimental {
                state.experimental_capital += capital;
            }

            state.estimated_daily_interactions += 1.0;
        }

        self.portfolio_state = state.clone();

        PortfolioDecision {
            selected_order_ids: selected,
            rejected_order_ids: rejected,
            reasons,
            state,
        }
    }

    pub async fn flush(&mut self, db: &DatabaseConnection) -> Result<(), Error> {
        for snapshot in self.pending_snapshots.drain(..) {
            LearningStore::insert_snapshot(db, &snapshot).await?;
            self.snapshots.push(snapshot);
        }

        while self.snapshots.len() > MAX_SNAPSHOTS_IN_MEMORY {
            self.snapshots.remove(0);
        }

        for (_, decision) in self.pending_decisions.drain() {
            LearningStore::upsert_decision(db, &decision).await?;
        }

        for outcome in self.pending_outcomes.drain(..) {
            LearningStore::insert_outcome(db, &outcome).await?;
        }

        for alert in self.pending_alerts.drain(..) {
            LearningStore::insert_alert(db, &alert).await?;
        }

        if self.config_dirty {
            LearningStore::save_config(db, &self.config).await?;
            self.config_dirty = false;
        }

        self.auto_tune();
        self.run_champion_challenger(db).await?;

        LearningStore::run_snapshot_retention(db, &self.config).await?;

        let inspector = self.build_inspector(db).await?;
        LearningStore::write_inspector(db, &inspector).await?;

        Ok(())
    }

    pub async fn build_inspector(
        &self,
        db: &DatabaseConnection,
    ) -> Result<LearningInspector, Error> {
        let completed: Vec<&DecisionRecord> = self
            .decisions
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

        let open = self
            .decisions
            .iter()
            .filter(|decision| {
                matches!(
                    decision.status,
                    DecisionStatus::Open | DecisionStatus::Partial | DecisionStatus::PaperOpen
                )
            })
            .count();

        let failed = self
            .decisions
            .iter()
            .filter(|decision| {
                matches!(
                    decision.status,
                    DecisionStatus::Expired | DecisionStatus::Rejected | DecisionStatus::Cancelled
                )
            })
            .count();

        let rewards: Vec<f64> = completed
            .iter()
            .filter_map(|decision| decision.actual_reward)
            .collect();

        let profits: Vec<f64> = completed
            .iter()
            .filter_map(|decision| decision.actual_profit)
            .collect();

        let prediction_errors: Vec<f64> = completed
            .iter()
            .filter_map(|decision| {
                decision
                    .actual_profit
                    .map(|actual| (actual - decision.predicted_profit).abs())
            })
            .collect();

        let fill_predictions: Vec<(f64, bool)> = completed
            .iter()
            .map(|decision| {
                (
                    decision.predicted_fill.fill_1h,
                    decision.actual_fill_hours.unwrap_or(f64::INFINITY) <= 1.0,
                )
            })
            .collect();

        let average_reward = mean(&rewards);
        let average_profit = mean(&profits);

        let shadow_names = [
            "shadow_conservative",
            "shadow_aggressive",
            "shadow_fast_turnover",
            "shadow_max_reward",
        ];

        let mut shadow_evaluations = Vec::new();
        for name in shadow_names {
            shadow_evaluations.push(offline_policy_evaluation(&self.decisions, name));
        }

        let alerts = LearningStore::load_alerts(db, 50).await?;
        let snapshots_recorded = LearningStore::count_snapshots(db).await?;

        let mut top_items: Vec<serde_json::Value> = self
            .learned
            .item_profiles
            .iter()
            .filter(|(_, profile)| profile.trade_count > 0)
            .map(|(key, profile)| {
                serde_json::json!({
                    "item_key": key,
                    "trades": profile.trade_count,
                    "confidence": profile.confidence,
                    "avg_profit": profile.avg_profit,
                    "avg_roi": profile.avg_roi,
                    "avg_hold_hours": profile.avg_hold_hours,
                    "avg_buy_fill_hours": profile.avg_buy_fill_hours,
                    "avg_sell_fill_hours": profile.avg_sell_fill_hours,
                    "sell_through": profile.sell_through,
                    "prediction_mae": profile.prediction_mae,
                    "prediction_accuracy": profile.prediction_accuracy,
                    "open_units": profile.open_units,
                    "avg_open_age_hours": profile.avg_open_age_hours,
                })
            })
            .collect();

        top_items.sort_by(|a, b| {
            let ac = a
                .get("confidence")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0);
            let bc = b
                .get("confidence")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0);
            bc.partial_cmp(&ac).unwrap_or(Ordering::Equal)
        });
        top_items.truncate(100);

        let attribution = profit_attribution(&completed);

        Ok(LearningInspector {
            model: MODEL_VERSION.to_string(),
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            reward_version: REWARD_VERSION,
            policy_version: POLICY_VERSION,
            updated_at: Some(Utc::now()),
            mode: self.config.mode,
            health: self.health.clone(),
            transactions_learned: self.learned.transaction_count,
            snapshots_recorded,
            decisions_total: self.decisions.len(),
            decisions_completed: completed.len(),
            decisions_failed: failed,
            decisions_open: open,
            champion: Some(ModelVersionRecord {
                id: "ultimate_v5_champion".into(),
                version: MODEL_VERSION.into(),
                role: ModelRole::Champion,
                active: !self.health.fallback_active,
                parameters: serde_json::to_value(&self.config).unwrap_or_default(),
                trained_on_decisions: self.decisions.len(),
                average_reward,
                failure_rate: failed as f64 / self.decisions.len().max(1) as f64,
                prediction_mae: mean(&prediction_errors),
                calibration_error: calibration_error(&fill_predictions),
                drawdown: maximum_drawdown(&rewards),
                created_at: Utc::now(),
                promoted_at: None,
            }),
            challengers: vec![],
            shadow_evaluations,
            metrics: EvaluationMetrics {
                sample_count: completed.len(),
                average_reward,
                median_reward: median(&rewards),
                failure_rate: failed as f64 / self.decisions.len().max(1) as f64,
                prediction_mae: mean(&prediction_errors),
                fill_brier_score: brier_score(&fill_predictions),
                calibration_error: calibration_error(&fill_predictions),
                max_drawdown: maximum_drawdown(&rewards),
                profit_per_hour: completed
                    .iter()
                    .filter_map(|decision| {
                        Some(decision.actual_profit? / decision.actual_cycle_hours?.max(0.05))
                    })
                    .sum::<f64>()
                    / completed.len().max(1) as f64,
                ..Default::default()
            },
            profit_attribution: attribution,
            feature_importance: feature_importance(&self.decisions),
            arbitrage_opportunities: self.arbitrage_opportunities.clone(),
            alerts,
            active_events: self.events.clone(),
            top_items,
            portfolio: self.portfolio_state.clone(),
            advanced: serde_json::to_value(advanced_diagnostics(
                &self.decisions,
                &self.snapshots,
                make_seed(&["algoframe", MODEL_VERSION, "advanced"]),
            ))
            .unwrap_or(serde_json::Value::Null),
            config: self.config.clone(),
        })
    }

    fn generate_buy_candidates(
        &self,
        context_key: &str,
        snapshot: &MarketSnapshot,
        price: &ItemPriceInfo,
        profile: &ItemProfile,
        minimum_profit: i64,
        minimum_margin_percent: i64,
        max_quantity: i64,
    ) -> Vec<CandidateAction> {
        let best_bid = snapshot.features.robust_bid.max(1);
        let market_ask = snapshot.features.robust_ask.max(best_bid + 1);
        let historical = historical_center(price);

        let live_weight = match snapshot.features.regime {
            MarketRegime::Normal => 0.88,
            MarketRegime::Moving => 0.94,
            MarketRegime::Shock => 0.98,
        };

        let mut expected_sell = market_ask as f64 * live_weight
            + historical.max(market_ask as f64) * (1.0 - live_weight);

        let profile_weight = profile.confidence
            * profile.prediction_accuracy
            * snapshot.features.regime.confidence_multiplier()
            * 0.10;

        if profile.avg_sale_price > 0.0 {
            expected_sell =
                expected_sell * (1.0 - profile_weight) + profile.avg_sale_price * profile_weight;
        }

        expected_sell *= 1.0
            + snapshot.features.event_signal.clamp(-0.20, 0.20) * 0.15
            + snapshot.features.forecast.opportunity_2h.clamp(-0.20, 0.20) * 0.10;

        let safe_by_profit = (expected_sell.floor() as i64 - minimum_profit).max(1);

        let safe_by_margin = if minimum_margin_percent <= -1 {
            safe_by_profit
        } else {
            let margin = minimum_margin_percent.max(0) as f64 / 100.0;
            (expected_sell / (1.0 + margin)).floor() as i64
        };

        let safe_max = safe_by_profit.min(safe_by_margin);
        let max_price = safe_max.min(best_bid + self.config.max_price_search_steps.max(0));

        if max_price < best_bid {
            return vec![];
        }

        let max_q = if self.config.quantity_optimization {
            max_quantity
        } else {
            1
        };

        let elasticity = self.elasticity_model(TradeSide::Buy, &snapshot.category);

        let mut candidates = Vec::new();

        let buy_price_candidates: Vec<i64> = if self.config.continuous_price_optimization {
            (best_bid..=max_price).collect()
        } else {
            [
                best_bid,
                best_bid.saturating_add(1),
                best_bid.saturating_add(2),
            ]
            .into_iter()
            .filter(|candidate| *candidate <= max_price)
            .collect()
        };

        for candidate_price in buy_price_candidates {
            for quantity in 1..=max_q {
                let offset = candidate_price - best_bid;
                let key = format!("buy:offset{offset:+}:q{quantity}");

                let evidence = self.hierarchical_evidence(
                    TradeSide::Buy,
                    context_key,
                    &snapshot.category,
                    &key,
                );

                let fallback_hours = (8.0 / (0.25 + snapshot.features.liquidity)).clamp(0.15, 96.0);

                let mut survival = if self.config.survival_models {
                    evidence.survival_prediction(fallback_hours)
                } else {
                    exponential_survival(fallback_hours)
                };

                let elasticity_multiplier =
                    elasticity.probability_multiplier(offset as f64, quantity);

                survival.fill_15m = (survival.fill_15m * elasticity_multiplier).clamp(0.01, 0.99);
                survival.fill_1h = (survival.fill_1h * elasticity_multiplier).clamp(0.01, 0.99);
                survival.fill_6h = (survival.fill_6h * elasticity_multiplier).clamp(0.01, 0.99);
                survival.fill_24h = (survival.fill_24h * elasticity_multiplier).clamp(0.01, 0.99);

                self.calibrate_survival(TradeSide::Buy, &snapshot.category, &key, &mut survival);

                survival.expected_fill_hours = (survival.expected_fill_hours
                    / elasticity_multiplier.max(0.25))
                .clamp(0.05, 720.0);

                self.calibrate_survival_category(TradeSide::Buy, &snapshot.category, &mut survival);

                let market_profit = expected_sell - candidate_price as f64;
                let prediction_correction = profile.prediction_bias
                    * profile.confidence
                    * profile.prediction_accuracy
                    * MAX_PREDICTION_BIAS_WEIGHT;

                let expected_profit = if market_profit > 0.0 {
                    (market_profit + prediction_correction)
                        .clamp(market_profit * 0.50, market_profit * 1.45 + 2.0)
                } else {
                    market_profit
                };

                let sell_hours = if profile.avg_hold_hours > 0.0 {
                    profile.avg_hold_hours
                } else if profile.avg_sell_fill_hours > 0.0 {
                    profile.avg_sell_fill_hours
                } else {
                    (16.0 / (0.25 + snapshot.features.liquidity)).clamp(0.25, 120.0)
                };

                let total_hours = survival.expected_fill_hours + sell_hours;
                let capital = candidate_price as f64 * quantity as f64;

                let mut reward = self.reward_breakdown(
                    TradeSide::Buy,
                    expected_profit,
                    candidate_price,
                    quantity,
                    total_hours,
                    snapshot,
                    profile,
                );

                let mut distributional_epistemic = 0.0;
                let mut distributional_aleatoric = 0.0;

                if self.config.distributional_predictions
                    || self.config.cvar_risk_optimization
                    || self.config.uncertainty_decomposition
                {
                    let (risk_multiplier, distribution) = distributional_risk_adjustment(
                        &self.decisions,
                        &snapshot.category,
                        expected_profit,
                    );

                    distributional_epistemic = distribution.epistemic_uncertainty;
                    distributional_aleatoric = distribution.aleatoric_uncertainty.clamp(0.0, 1.0);

                    if self.config.cvar_risk_optimization {
                        reward.final_reward *= risk_multiplier;
                        reward.risk_penalty += expected_profit.max(0.0)
                            * (1.0 - risk_multiplier)
                            * self.config.downside_risk_weight;
                    }
                }

                let uncertainty =
                    (1.0 - profile.confidence * 0.50 - snapshot.features.quality.score * 0.30
                        + snapshot.features.anomaly.score * 0.30
                        + distributional_epistemic * self.config.epistemic_risk_weight
                        + distributional_aleatoric * self.config.aleatoric_risk_weight)
                        .clamp(0.0, 1.0);

                let required_roi = if minimum_margin_percent <= -1 {
                    0.0
                } else {
                    minimum_margin_percent.max(0) as f64 / 100.0
                };

                let roi = expected_profit / candidate_price.max(1) as f64;

                let mut reject_reasons = Vec::new();
                if expected_profit < minimum_profit as f64 {
                    reject_reasons.push("below minimum expected profit".into());
                }
                if roi < required_roi {
                    reject_reasons.push("below minimum ROI".into());
                }

                candidates.push(CandidateAction {
                    key,
                    side: TradeSide::Buy,
                    price: candidate_price,
                    quantity,
                    price_offset: offset,
                    expected_sell_price: expected_sell,
                    survival,
                    reward,
                    bandit_sample: 0.0,
                    propensity: 0.0,
                    uncertainty,
                    valid: reject_reasons.is_empty() && capital > 0.0,
                    reject_reasons,
                });
            }
        }

        candidates
    }

    fn generate_sell_candidates(
        &self,
        context_key: &str,
        snapshot: &MarketSnapshot,
        _price: &ItemPriceInfo,
        profile: &ItemProfile,
        minimum_profit: i64,
        max_quantity: i64,
        bought_price: i64,
        current_order_price: i64,
    ) -> Vec<CandidateAction> {
        let market_floor = snapshot.features.robust_ask.max(1);
        let minimum_allowed = bought_price.saturating_add(minimum_profit).max(1);

        let lower = minimum_allowed
            .max(market_floor.saturating_sub(self.config.max_price_search_steps.max(0)));

        let upper = market_floor
            .saturating_add(3)
            .max(current_order_price)
            .max(lower);

        let elasticity = self.elasticity_model(TradeSide::Sell, &snapshot.category);

        let max_q = if self.config.quantity_optimization {
            max_quantity
        } else {
            1
        };

        let mut candidates = Vec::new();

        let sell_price_candidates: Vec<i64> = if self.config.continuous_price_optimization {
            (lower..=upper).collect()
        } else {
            [
                current_order_price.max(minimum_allowed),
                market_floor.max(minimum_allowed),
                market_floor.saturating_sub(1).max(minimum_allowed),
                market_floor.saturating_sub(2).max(minimum_allowed),
            ]
            .into_iter()
            .filter(|candidate| *candidate >= lower && *candidate <= upper)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
        };

        for candidate_price in sell_price_candidates {
            for quantity in 1..=max_q {
                let advantage = market_floor - candidate_price;
                let key = format!("sell:offset{advantage:+}:q{quantity}");

                let evidence = self.hierarchical_evidence(
                    TradeSide::Sell,
                    context_key,
                    &snapshot.category,
                    &key,
                );

                let fallback_hours = if profile.avg_sell_fill_hours > 0.0 {
                    profile.avg_sell_fill_hours
                } else {
                    (12.0 / (0.25 + snapshot.features.liquidity)).clamp(0.15, 120.0)
                };

                let mut survival = if self.config.survival_models {
                    evidence.survival_prediction(fallback_hours)
                } else {
                    exponential_survival(fallback_hours)
                };

                let elasticity_multiplier =
                    elasticity.probability_multiplier(advantage as f64, quantity);

                survival.fill_15m = (survival.fill_15m * elasticity_multiplier).clamp(0.01, 0.99);
                survival.fill_1h = (survival.fill_1h * elasticity_multiplier).clamp(0.01, 0.99);
                survival.fill_6h = (survival.fill_6h * elasticity_multiplier).clamp(0.01, 0.99);
                survival.fill_24h = (survival.fill_24h * elasticity_multiplier).clamp(0.01, 0.99);

                self.calibrate_survival(TradeSide::Sell, &snapshot.category, &key, &mut survival);

                survival.expected_fill_hours = (survival.expected_fill_hours
                    / elasticity_multiplier.max(0.25))
                .clamp(0.05, 720.0);

                self.calibrate_survival_category(
                    TradeSide::Sell,
                    &snapshot.category,
                    &mut survival,
                );

                let expected_profit = (candidate_price - bought_price).max(0) as f64;

                let mut reward = self.reward_breakdown(
                    TradeSide::Sell,
                    expected_profit,
                    bought_price.max(1),
                    quantity,
                    survival.expected_fill_hours,
                    snapshot,
                    profile,
                );

                let mut distributional_epistemic = 0.0;
                let mut distributional_aleatoric = 0.0;

                if self.config.distributional_predictions
                    || self.config.cvar_risk_optimization
                    || self.config.uncertainty_decomposition
                {
                    let (risk_multiplier, distribution) = distributional_risk_adjustment(
                        &self.decisions,
                        &snapshot.category,
                        expected_profit,
                    );

                    distributional_epistemic = distribution.epistemic_uncertainty;
                    distributional_aleatoric = distribution.aleatoric_uncertainty.clamp(0.0, 1.0);

                    if self.config.cvar_risk_optimization {
                        reward.final_reward *= risk_multiplier;
                        reward.risk_penalty += expected_profit.max(0.0)
                            * (1.0 - risk_multiplier)
                            * self.config.downside_risk_weight;
                    }
                }

                let uncertainty =
                    (1.0 - profile.confidence * 0.45 - snapshot.features.quality.score * 0.35
                        + snapshot.features.anomaly.score * 0.35
                        + distributional_epistemic * self.config.epistemic_risk_weight
                        + distributional_aleatoric * self.config.aleatoric_risk_weight)
                        .clamp(0.0, 1.0);

                let mut reject_reasons = Vec::new();

                if expected_profit < minimum_profit as f64 {
                    reject_reasons.push("below minimum expected profit".into());
                }

                candidates.push(CandidateAction {
                    key,
                    side: TradeSide::Sell,
                    price: candidate_price,
                    quantity,
                    price_offset: advantage,
                    expected_sell_price: candidate_price as f64,
                    survival,
                    reward,
                    bandit_sample: 0.0,
                    propensity: 0.0,
                    uncertainty,
                    valid: reject_reasons.is_empty(),
                    reject_reasons,
                });
            }
        }

        candidates
    }

    fn reward_breakdown(
        &self,
        side: TradeSide,
        expected_profit_per_unit: f64,
        capital_per_unit: i64,
        quantity: i64,
        hours: f64,
        snapshot: &MarketSnapshot,
        profile: &ItemProfile,
    ) -> RewardBreakdown {
        let gross_profit = expected_profit_per_unit * quantity as f64;
        let capital = capital_per_unit.max(1) as f64 * quantity as f64;

        let human_minutes = self.config.human_minutes_per_trade;
        let human_cost = human_minutes / 60.0 * self.config.human_time_value_plat_per_hour;

        let risk_penalty = gross_profit.max(1.0)
            * (snapshot.features.volatility * 0.08 + snapshot.features.anomaly.score * 0.20)
                .clamp(0.0, 0.60);

        let inventory_penalty = if side == TradeSide::Buy {
            gross_profit.max(1.0) * snapshot.features.inventory_pressure * 0.30
        } else {
            0.0
        };

        let opportunity_cost_penalty = if side == TradeSide::Buy {
            gross_profit.max(1.0) * snapshot.features.opportunity_cost_signal.max(0.0) * 0.10
        } else {
            0.0
        };

        let arbitrage_bonus =
            gross_profit.max(0.0) * snapshot.features.arbitrage_signal.max(0.0) * 0.20;

        let event_adjustment =
            gross_profit * snapshot.features.event_signal.clamp(-1.0, 1.0) * 0.10;

        let net_profit = gross_profit - risk_penalty - inventory_penalty - opportunity_cost_penalty
            + arbitrage_bonus
            + event_adjustment;

        let expected_roi = if capital > 0.0 {
            net_profit / capital
        } else {
            0.0
        };

        let expected_platinum_per_hour = net_profit / hours.max(0.05);

        let mut final_reward = normalized_reward(
            net_profit,
            capital,
            hours.max(0.05),
            human_minutes,
            self.config.human_time_value_plat_per_hour,
        );

        final_reward *= match self.config.mode {
            OperatingMode::Paper => 1.0,
            OperatingMode::Conservative => {
                (1.0 - snapshot.features.volatility * 0.12 - snapshot.features.anomaly.score * 0.18)
                    .clamp(0.45, 1.0)
            }
            OperatingMode::Balanced => 1.0,
            OperatingMode::Growth => (1.0 + expected_roi.max(0.0) * 0.08).clamp(1.0, 1.15),
            OperatingMode::Liquid => {
                (1.0 + (1.0 / hours.max(0.25)).min(2.0) * 0.10).clamp(1.0, 1.20)
            }
        };

        RewardBreakdown {
            expected_profit: expected_profit_per_unit,
            expected_roi,
            expected_platinum_per_hour,
            capital_cost: capital,
            human_time_cost: human_cost,
            risk_penalty,
            inventory_penalty,
            anomaly_penalty: risk_penalty * snapshot.features.anomaly.score,
            opportunity_cost_penalty,
            arbitrage_bonus,
            event_adjustment,
            final_reward: final_reward * (0.50 + 0.50 * profile.profit_stability.max(0.25)),
        }
    }

    fn apply_candidate_guardrails(
        &self,
        candidate: &mut CandidateAction,
        snapshot: &MarketSnapshot,
        capital_budget: f64,
        profile: &ItemProfile,
        fallback_active: bool,
    ) {
        let capital = match candidate.side {
            TradeSide::Buy => candidate.price as f64 * candidate.quantity as f64,
            TradeSide::Sell => candidate.reward.capital_cost,
        };

        if candidate.side == TradeSide::Buy
            && capital > capital_budget * self.config.max_single_decision_pct
        {
            candidate.valid = false;
            candidate
                .reject_reasons
                .push("single-decision capital limit".into());
        }

        if candidate.side == TradeSide::Buy {
            let confidence_cap = if candidate.uncertainty >= 0.70 {
                1
            } else if candidate.uncertainty >= 0.50 {
                2
            } else if candidate.uncertainty >= 0.30 {
                3
            } else {
                self.config.max_trade_quantity
            };

            if candidate.quantity > confidence_cap {
                candidate.valid = false;
                candidate.reject_reasons.push(format!(
                    "uncertainty-aware position size cap ({})",
                    confidence_cap
                ));
            }
        }

        if snapshot.features.inventory_pressure > self.config.max_inventory_pressure
            && candidate.side == TradeSide::Buy
        {
            candidate.valid = false;
            candidate
                .reject_reasons
                .push("inventory pressure hard limit".into());
        }

        if snapshot.features.anomaly.score >= 0.85 {
            candidate.valid = false;
            candidate
                .reject_reasons
                .push("market anomaly hard stop".into());
        }

        if snapshot.features.quality.score < self.config.minimum_data_quality {
            candidate.valid = false;
            candidate
                .reject_reasons
                .push("insufficient data quality".into());
        }

        if fallback_active {
            if candidate.quantity > 1 {
                candidate.valid = false;
                candidate
                    .reject_reasons
                    .push("fallback limits quantity to one".into());
            }

            if candidate.uncertainty > 0.40 {
                candidate.valid = false;
                candidate
                    .reject_reasons
                    .push("fallback rejects high uncertainty".into());
            }
        }

        match self.config.mode {
            OperatingMode::Conservative => {
                if candidate.uncertainty > 0.45 || snapshot.features.volatility > 0.65 {
                    candidate.valid = false;
                    candidate
                        .reject_reasons
                        .push("conservative mode risk limit".into());
                }
            }
            OperatingMode::Liquid => {
                if candidate.side == TradeSide::Buy && candidate.survival.fill_1h < 0.55 {
                    candidate.valid = false;
                    candidate
                        .reject_reasons
                        .push("liquid mode requires fast fill".into());
                }
            }
            OperatingMode::Growth => {}
            OperatingMode::Balanced => {}
            OperatingMode::Paper => {}
        }

        if profile.trade_count == 0 && candidate.side == TradeSide::Buy && candidate.quantity > 1 {
            candidate.valid = false;
            candidate
                .reject_reasons
                .push("cold-start exploration limited to quantity one".into());
        }

        if self.config.active_learning
            && candidate.side == TradeSide::Buy
            && candidate.uncertainty >= 0.65
            && candidate.quantity > 1
        {
            candidate.valid = false;
            candidate
                .reject_reasons
                .push("active-learning exploration is limited to quantity one".into());
        }

        let today = Utc::now().date_naive();
        let interactions_today = self
            .decisions
            .iter()
            .filter(|decision| decision.created_at.date_naive() == today)
            .filter(|decision| decision.status != DecisionStatus::Rejected)
            .count() as i64;

        if interactions_today >= self.config.max_daily_trade_interactions {
            candidate.valid = false;
            candidate
                .reject_reasons
                .push("daily human trade-capacity limit".into());
        }
    }

    fn hierarchical_evidence(
        &self,
        side: TradeSide,
        context_key: &str,
        category: &str,
        action_key: &str,
    ) -> BanditEvidence {
        let exact = self
            .bandit
            .get(&(side, context_key.to_string(), action_key.to_string()))
            .cloned()
            .unwrap_or_default();

        if !self.config.hierarchical_learning {
            return exact;
        }

        let category_prefix = format!("{}|{}|", side_name(side), category);

        let family_marker = context_key
            .split("|family:")
            .nth(1)
            .and_then(|rest| rest.split("|item:").next())
            .map(|family| format!("|family:{}|", family));

        let family_evidence: Vec<BanditEvidence> = self
            .bandit
            .iter()
            .filter(|((entry_side, context, action), _)| {
                *entry_side == side
                    && action == action_key
                    && family_marker
                        .as_ref()
                        .map(|marker| context.contains(marker))
                        .unwrap_or(false)
            })
            .map(|(_, evidence)| evidence.clone())
            .collect();

        let category_evidence: Vec<BanditEvidence> = self
            .bandit
            .iter()
            .filter(|((entry_side, context, action), _)| {
                *entry_side == side && context.starts_with(&category_prefix) && action == action_key
            })
            .map(|(_, evidence)| evidence.clone())
            .collect();

        let global_evidence: Vec<BanditEvidence> = self
            .bandit
            .iter()
            .filter(|((entry_side, _, action), _)| *entry_side == side && action == action_key)
            .map(|(_, evidence)| evidence.clone())
            .collect();

        let family = merge_evidence_many(&family_evidence, 0.55);
        let category = merge_evidence_many(&category_evidence, 0.28);
        let global = merge_evidence_many(&global_evidence, 0.10);

        merge_evidence(
            &merge_evidence(&exact, &family, &BanditEvidence::default()),
            &category,
            &global,
        )
    }

    fn calibrate_survival(
        &self,
        side: TradeSide,
        category: &str,
        action_key: &str,
        survival: &mut SurvivalPrediction,
    ) {
        let relevant: Vec<&DecisionRecord> = self
            .decisions
            .iter()
            .rev()
            .filter(|decision| {
                decision.side == side
                    && decision.category == category
                    && decision.chosen_action == action_key
                    && matches!(
                        decision.status,
                        DecisionStatus::Completed
                            | DecisionStatus::Filled
                            | DecisionStatus::Expired
                            | DecisionStatus::PaperCompleted
                    )
            })
            .take(200)
            .collect();

        if relevant.len() < 8 {
            survival.calibration = 0.50;
            return;
        }

        fn ratio(
            relevant: &[&DecisionRecord],
            predicted: fn(&SurvivalPrediction) -> f64,
            horizon: f64,
        ) -> (f64, f64) {
            let predicted_mean = relevant
                .iter()
                .map(|decision| predicted(&decision.predicted_fill))
                .sum::<f64>()
                / relevant.len() as f64;

            let actual_mean = relevant
                .iter()
                .filter(|decision| decision.actual_fill_hours.unwrap_or(f64::INFINITY) <= horizon)
                .count() as f64
                / relevant.len() as f64;

            let multiplier = if predicted_mean > 0.01 {
                (actual_mean / predicted_mean).clamp(0.50, 1.50)
            } else {
                1.0
            };

            (multiplier, (actual_mean - predicted_mean).abs())
        }

        let (m15, e15) = ratio(&relevant, |value| value.fill_15m, 0.25);
        let (m1, e1) = ratio(&relevant, |value| value.fill_1h, 1.0);
        let (m6, e6) = ratio(&relevant, |value| value.fill_6h, 6.0);
        let (m24, e24) = ratio(&relevant, |value| value.fill_24h, 24.0);

        survival.fill_15m = (survival.fill_15m * m15).clamp(0.01, 0.99);
        survival.fill_1h = (survival.fill_1h * m1).clamp(0.01, 0.99);
        survival.fill_6h = (survival.fill_6h * m6).clamp(0.01, 0.99);
        survival.fill_24h = (survival.fill_24h * m24).clamp(0.01, 0.99);
        survival.calibration = (1.0 - (e15 + e1 + e6 + e24) / 4.0).clamp(0.0, 1.0);
    }

    fn calibrate_survival_category(
        &self,
        side: TradeSide,
        category: &str,
        survival: &mut SurvivalPrediction,
    ) {
        let completed: Vec<&DecisionRecord> = self
            .decisions
            .iter()
            .filter(|decision| {
                decision.side == side
                    && decision.category == category
                    && matches!(
                        decision.status,
                        DecisionStatus::Completed
                            | DecisionStatus::Filled
                            | DecisionStatus::PaperCompleted
                            | DecisionStatus::Expired
                    )
            })
            .collect();

        if completed.len() < 12 {
            survival.calibration = 0.25;
            return;
        }

        let calibrate = |current: f64, horizon: f64, selector: fn(&SurvivalPrediction) -> f64| {
            let mut successes = 0.0;
            let mut total = 0.0;

            for decision in &completed {
                let predicted = selector(&decision.predicted_fill);
                if (predicted - current).abs() > 0.18 {
                    continue;
                }

                let weight =
                    recency_weight(decision.updated_at, self.config.recency_half_life_days);

                let observed = decision
                    .actual_fill_hours
                    .map(|hours| hours <= horizon)
                    .unwrap_or(false);

                total += weight;
                if observed {
                    successes += weight;
                }
            }

            if total < 4.0 {
                return (current, 0.0);
            }

            // Beta-style shrinkage back toward the model probability prevents
            // small calibration bins from over-correcting.
            let empirical = (successes + current * 4.0) / (total + 4.0);
            let confidence = (total / (total + 12.0)).clamp(0.0, 0.85);

            (
                (current * (1.0 - confidence) + empirical * confidence).clamp(0.01, 0.99),
                confidence,
            )
        };

        let (p15, c15) = calibrate(survival.fill_15m, 0.25, |prediction| prediction.fill_15m);
        let (p1h, c1h) = calibrate(survival.fill_1h, 1.0, |prediction| prediction.fill_1h);
        let (p6h, c6h) = calibrate(survival.fill_6h, 6.0, |prediction| prediction.fill_6h);
        let (p24h, c24h) = calibrate(survival.fill_24h, 24.0, |prediction| prediction.fill_24h);

        // Preserve probability monotonicity after independent calibration.
        survival.fill_15m = p15;
        survival.fill_1h = p1h.max(survival.fill_15m);
        survival.fill_6h = p6h.max(survival.fill_1h);
        survival.fill_24h = p24h.max(survival.fill_6h);
        survival.calibration = mean(&[c15, c1h, c6h, c24h]).clamp(0.0, 1.0);
    }

    fn elasticity_model(&self, side: TradeSide, category: &str) -> ElasticityModel {
        let records: Vec<(f64, f64, bool, f64)> = self
            .decisions
            .iter()
            .filter(|decision| {
                decision.side == side
                    && decision.category == category
                    && matches!(
                        decision.status,
                        DecisionStatus::Completed
                            | DecisionStatus::Expired
                            | DecisionStatus::PaperCompleted
                    )
            })
            .map(|decision| {
                let advantage = match side {
                    TradeSide::Buy => decision.price - decision.features.robust_bid,
                    TradeSide::Sell => decision.features.robust_ask - decision.price,
                } as f64;

                let filled = !matches!(decision.status, DecisionStatus::Expired);

                (
                    advantage,
                    decision.quantity as f64,
                    filled,
                    recency_weight(decision.updated_at, self.config.recency_half_life_days),
                )
            })
            .collect();

        ElasticityModel::fit(&records)
    }

    fn contextual_profile(&self, item_key: &str, category: &str) -> ItemProfile {
        let item = self
            .learned
            .item_profiles
            .get(item_key)
            .cloned()
            .unwrap_or_default();

        if !self.config.hierarchical_learning {
            return item;
        }

        let category_profile = self
            .learned
            .category_profiles
            .get(category)
            .cloned()
            .unwrap_or_else(|| self.learned.global.clone());

        let mut profile = blend_profile(&category_profile, &item, item.confidence);

        profile.purchase_units = item.purchase_units;
        profile.sold_units = item.sold_units;
        profile.matched_units = item.matched_units;
        profile.open_units = item.open_units;
        profile.trade_count = item.trade_count;
        profile.confidence = item.confidence;

        profile
    }

    fn item_lifecycle(&self, item: &CacheTradableItem, profile: &ItemProfile) -> ItemLifecycle {
        let name = item.name.to_lowercase();
        let tags: Vec<String> = item.tags.iter().map(|tag| tag.to_lowercase()).collect();
        let is_prime = name.contains("prime") || tags.iter().any(|tag| tag.contains("prime"));

        let matching_events: Vec<&EventSignal> = self
            .events
            .iter()
            .filter(|event| {
                event.keywords.iter().any(|keyword| {
                    let keyword = keyword.to_lowercase();
                    name.contains(&keyword) || tags.iter().any(|tag| tag.contains(&keyword))
                })
            })
            .collect();

        if matching_events.iter().any(|event| {
            event.kind == "prime_resurgence" || event.title.to_lowercase().contains("resurgence")
        }) {
            return ItemLifecycle::Resurgence;
        }

        if matching_events.iter().any(|event| {
            event.impact > 0.20
                && (event.title.to_lowercase().contains("vault")
                    || event.title.to_lowercase().contains("retir"))
        }) {
            return ItemLifecycle::RecentlyVaulted;
        }

        if matching_events.iter().any(|event| {
            let title = event.title.to_lowercase();
            title.contains("prime access")
                || title.contains("new prime")
                || title.contains("arrives")
        }) {
            return ItemLifecycle::NewRelease;
        }

        if !matching_events.is_empty() {
            return ItemLifecycle::EventAffected;
        }

        if is_prime && profile.trade_count >= 20 {
            ItemLifecycle::Mature
        } else if is_prime && profile.trade_count < 5 {
            ItemLifecycle::NewRelease
        } else {
            ItemLifecycle::Standard
        }
    }

    fn event_signal(&self, item: &CacheTradableItem) -> f64 {
        let now = Utc::now();
        let name = item.name.to_lowercase();
        let tags: Vec<String> = item.tags.iter().map(|tag| tag.to_lowercase()).collect();

        self.events
            .iter()
            .filter(|event| {
                event.starts_at <= now && event.ends_at.map(|end| end >= now).unwrap_or(true)
            })
            .filter(|event| {
                event.keywords.iter().any(|keyword| {
                    let keyword = keyword.to_lowercase();
                    name.contains(&keyword) || tags.iter().any(|tag| tag.contains(&keyword))
                }) || event.tags.iter().any(|event_tag| {
                    let event_tag = event_tag.to_lowercase();
                    tags.iter().any(|tag| tag == &event_tag)
                })
            })
            .map(|event| event.impact * event.confidence)
            .sum::<f64>()
            .clamp(-1.0, 1.0)
    }

    fn market_wide_opportunity_cost(&self) -> f64 {
        let recent: Vec<f64> = self
            .snapshots
            .iter()
            .rev()
            .take(100)
            .map(|snapshot| snapshot.features.forecast.opportunity_2h.max(0.0))
            .collect();

        mean(&recent).clamp(0.0, 1.0)
    }

    fn arbitrage_signal(&self, item_key: &str, item_name: &str) -> f64 {
        let family = family_from_name(item_name);

        self.arbitrage_opportunities
            .iter()
            .filter(|opportunity| {
                opportunity.family == family || opportunity.legs.iter().any(|leg| leg == item_key)
            })
            .map(|opportunity| opportunity.score)
            .fold(0.0_f64, f64::max)
            .clamp(0.0, 1.0)
    }

    fn discover_arbitrage(&self) -> Vec<ArbitrageOpportunity> {
        let mut latest: HashMap<String, &MarketSnapshot> = HashMap::new();

        for snapshot in self.snapshots.iter().chain(self.pending_snapshots.iter()) {
            latest.insert(snapshot.item_key.clone(), snapshot);
        }

        let mut by_family: HashMap<String, Vec<&MarketSnapshot>> = HashMap::new();

        for snapshot in latest.values() {
            by_family
                .entry(family_from_name(&snapshot.item_name))
                .or_default()
                .push(*snapshot);
        }

        let mut opportunities = Vec::new();

        for (family, members) in by_family {
            let set = members
                .iter()
                .find(|snapshot| snapshot.item_name.to_lowercase().ends_with(" set"));

            let Some(set) = set else {
                continue;
            };

            let components: Vec<&&MarketSnapshot> = members
                .iter()
                .filter(|snapshot| snapshot.item_key != set.item_key)
                .filter(|snapshot| snapshot.features.robust_ask > 0)
                .collect();

            if components.len() < 2 || set.features.robust_bid <= 0 {
                continue;
            }

            let cost = components
                .iter()
                .map(|snapshot| snapshot.features.robust_ask as f64)
                .sum::<f64>();

            let revenue = set.features.robust_bid as f64;
            let profit = revenue - cost;

            if profit <= 0.0 {
                continue;
            }

            let score = (profit / cost.max(1.0)).clamp(0.0, 1.0);

            opportunities.push(ArbitrageOpportunity {
                id: format!("set:{family}"),
                kind: "set_assembly".into(),
                family: family.clone(),
                legs: components
                    .iter()
                    .map(|snapshot| snapshot.item_key.clone())
                    .chain(std::iter::once(set.item_key.clone()))
                    .collect(),
                cost,
                expected_revenue: revenue,
                expected_profit: profit,
                expected_hours: 4.0,
                score,
                confidence: (components
                    .iter()
                    .map(|snapshot| snapshot.features.quality.score)
                    .sum::<f64>()
                    / components.len() as f64)
                    .clamp(0.0, 1.0),
            });
        }

        // Automatic Arcane rank arbitrage: rank N requires the triangular
        // number of unranked copies (R0=1, R1=3, ... R5=21).
        let mut arcanes_by_wfm: HashMap<String, Vec<&MarketSnapshot>> = HashMap::new();

        for snapshot in latest.values() {
            if snapshot.category == "arcane" {
                arcanes_by_wfm
                    .entry(snapshot.wfm_id.clone())
                    .or_default()
                    .push(*snapshot);
            }
        }

        for (wfm_id, variants) in arcanes_by_wfm {
            let rank_of = |snapshot: &&MarketSnapshot| -> i64 {
                snapshot
                    .sub_type
                    .get("rank")
                    .and_then(|value| value.as_i64())
                    .unwrap_or(0)
            };

            let unranked = variants
                .iter()
                .filter(|snapshot| rank_of(snapshot) == 0)
                .min_by_key(|snapshot| snapshot.features.robust_ask);

            let Some(unranked) = unranked else {
                continue;
            };

            if unranked.features.robust_ask <= 0 {
                continue;
            }

            for ranked in variants.iter().filter(|snapshot| rank_of(snapshot) > 0) {
                let rank = rank_of(ranked);
                let copies = ((rank + 1) * (rank + 2) / 2).max(1);
                let cost = unranked.features.robust_ask as f64 * copies as f64;
                let revenue = ranked.features.robust_bid as f64;

                if revenue <= cost || revenue <= 0.0 {
                    continue;
                }

                let profit = revenue - cost;

                opportunities.push(ArbitrageOpportunity {
                    id: format!("arcane:{wfm_id}:rank{rank}"),
                    kind: "arcane_rank".into(),
                    family: ranked.item_name.to_lowercase(),
                    legs: vec![unranked.item_key.clone(), ranked.item_key.clone()],
                    cost,
                    expected_revenue: revenue,
                    expected_profit: profit,
                    expected_hours: 4.0,
                    score: (profit / cost.max(1.0)).clamp(0.0, 1.0),
                    confidence: unranked
                        .features
                        .quality
                        .score
                        .min(ranked.features.quality.score),
                });
            }
        }

        // Relic expected-value arbitrage. This uses DE-derived public drop
        // tables and only values rewards for which AlgoFrame has a current market
        // snapshot, so it never invents a platinum value.
        if !self.relic_tables.is_empty() {
            let mut latest_by_name: HashMap<String, &MarketSnapshot> = HashMap::new();

            for snapshot in latest.values() {
                latest_by_name.insert(snapshot.item_name.to_lowercase(), *snapshot);
            }

            for relic in latest.values() {
                let Some((tier, relic_name)) = parse_relic_identity(&relic.item_name) else {
                    continue;
                };

                if relic.features.robust_ask <= 0 {
                    continue;
                }

                let refinement = normalized_refinement(
                    relic
                        .sub_type
                        .get("variant")
                        .and_then(|value| value.as_str()),
                );

                let Some(table) = self.relic_tables.iter().find(|table| {
                    table.tier.eq_ignore_ascii_case(&tier)
                        && table.relic_name.eq_ignore_ascii_case(&relic_name)
                        && table.state.eq_ignore_ascii_case(&refinement)
                }) else {
                    continue;
                };

                let mut expected_revenue = 0.0;
                let mut covered_chance = 0.0;
                let mut reward_legs = Vec::new();
                let mut reward_quality = Vec::new();

                for reward in &table.rewards {
                    let Some(reward_snapshot) =
                        latest_by_name.get(&reward.item_name.to_lowercase())
                    else {
                        continue;
                    };

                    let sale_value = reward_snapshot.features.robust_bid;
                    if sale_value <= 0 {
                        continue;
                    }

                    let probability = (reward.chance / 100.0).clamp(0.0, 1.0);
                    expected_revenue += sale_value as f64 * probability;
                    covered_chance += probability;
                    reward_legs.push(reward_snapshot.item_key.clone());
                    reward_quality.push(reward_snapshot.features.quality.score);
                }

                if covered_chance < 0.70 {
                    continue;
                }

                let cost = relic.features.robust_ask as f64;
                let expected_profit = expected_revenue - cost;

                if expected_profit <= 0.0 {
                    continue;
                }

                let coverage_confidence = covered_chance.clamp(0.0, 1.0);
                let average_reward_quality = mean(&reward_quality);
                let confidence = (relic.features.quality.score * 0.35
                    + average_reward_quality * 0.35
                    + coverage_confidence * 0.30)
                    .clamp(0.0, 1.0);

                let mut legs = vec![relic.item_key.clone()];
                legs.extend(reward_legs);

                opportunities.push(ArbitrageOpportunity {
                    id: format!(
                        "relic:{}:{}:{}",
                        tier.to_lowercase(),
                        relic_name.to_lowercase(),
                        refinement.to_lowercase()
                    ),
                    kind: "relic_expected_value".into(),
                    family: format!("{} {} relic", tier, relic_name).to_lowercase(),
                    legs,
                    cost,
                    expected_revenue,
                    expected_profit,
                    expected_hours: 0.20,
                    score: ((expected_profit / cost.max(1.0)) * confidence).clamp(0.0, 1.0),
                    confidence,
                });
            }
        }

        // Explicit conversion graph supports relic/drop, crafting, refinement,
        // mod-rank and any custom multi-step relationship that cannot be inferred
        // reliably from names alone.
        let mut by_target: HashMap<String, Vec<&GraphEdge>> = HashMap::new();
        for edge in &self.graph_edges {
            by_target.entry(edge.to_key.clone()).or_default().push(edge);
        }

        for (target, edges) in by_target {
            let Some(target_snapshot) = latest.get(&target) else {
                continue;
            };

            let mut cost = 0.0;
            let mut legs = Vec::new();
            let mut valid = true;

            for edge in edges {
                let Some(source) = latest.get(&edge.from_key) else {
                    valid = false;
                    break;
                };

                if source.features.robust_ask <= 0 {
                    valid = false;
                    break;
                }

                cost += source.features.robust_ask as f64 * edge.quantity + edge.cost;
                legs.push(edge.from_key.clone());
            }

            if !valid || cost <= 0.0 || target_snapshot.features.robust_bid <= 0 {
                continue;
            }

            let revenue = target_snapshot.features.robust_bid as f64;
            let profit = revenue - cost;

            if profit <= 0.0 {
                continue;
            }

            legs.push(target.clone());

            opportunities.push(ArbitrageOpportunity {
                id: format!("graph:{target}"),
                kind: "conversion_graph".into(),
                family: target.clone(),
                legs,
                cost,
                expected_revenue: revenue,
                expected_profit: profit,
                expected_hours: 6.0,
                score: (profit / cost).clamp(0.0, 1.0),
                confidence: target_snapshot.features.quality.score,
            });
        }

        opportunities.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        opportunities.truncate(100);
        opportunities
    }

    fn reconcile_outcomes(&mut self, transactions: &[TransactionModel]) {
        let now = Utc::now();
        let mut decision_index: HashMap<String, usize> = self
            .decisions
            .iter()
            .enumerate()
            .map(|(index, decision)| (decision.id.clone(), index))
            .collect();

        // Direct transaction-to-order linkage handles partial fills naturally.
        for transaction in transactions {
            let Some(decision_id) =
                transaction_property_string(transaction, "algoframe_decision_id")
            else {
                continue;
            };

            let Some(index) = decision_index.get(&decision_id).copied() else {
                continue;
            };

            let decision = &mut self.decisions[index];

            if transaction.created_at < decision.created_at {
                continue;
            }

            decision.filled_quantity =
                (decision.filled_quantity + transaction.quantity.max(0)).min(decision.quantity);

            decision.fill_ratio = decision.filled_quantity as f64 / decision.quantity.max(1) as f64;

            let fill_hours = (transaction.created_at - decision.created_at)
                .num_seconds()
                .max(0) as f64
                / 3600.0;

            decision.actual_fill_hours = Some(
                decision
                    .actual_fill_hours
                    .map(|previous| previous.min(fill_hours))
                    .unwrap_or(fill_hours),
            );
            decision.filled_at = Some(transaction.created_at);
            decision.updated_at = transaction.created_at;

            if decision.fill_ratio < 1.0 {
                decision.status = DecisionStatus::Partial;
                decision.lifecycle = LifecycleState::PartiallyFilled;
            } else if decision.side == TradeSide::Buy {
                decision.status = DecisionStatus::Filled;
                decision.lifecycle = LifecycleState::Purchased;
            } else {
                decision.status = DecisionStatus::Completed;
                decision.lifecycle = LifecycleState::Sold;
                decision.completed_at = Some(transaction.created_at);

                let actual_profit = transaction
                    .profit
                    .map(|profit| profit as f64 / transaction.quantity.max(1) as f64)
                    .unwrap_or(decision.predicted_profit);

                let reward = normalized_reward(
                    actual_profit * transaction.quantity.max(1) as f64,
                    decision.capital.max(1.0),
                    fill_hours.max(0.05),
                    self.config.human_minutes_per_trade,
                    self.config.human_time_value_plat_per_hour,
                );

                decision.actual_profit = Some(actual_profit);
                decision.actual_reward = Some(reward);
                decision.actual_cycle_hours = Some(fill_hours);

                self.pending_outcomes.push(OutcomeRecord {
                    id: format!("{}:sale", decision.id),
                    decision_id: decision.id.clone(),
                    item_key: decision.item_key.clone(),
                    side: decision.side,
                    outcome_type: "sale_completed".into(),
                    quantity: transaction.quantity,
                    profit: actual_profit,
                    reward,
                    fill_hours,
                    cycle_hours: fill_hours,
                    competition_response: decision.features.dynamics.competitive_response_score,
                    simulated: matches!(decision.status, DecisionStatus::PaperCompleted),
                    created_at: transaction.created_at,
                });
            }

            self.pending_decisions
                .insert(decision.id.clone(), decision.clone());
        }

        // FIFO purchase -> sale matching gives buy actions their true end-to-end reward.
        let mut lots_by_item: HashMap<String, VecDeque<PurchaseLot>> = HashMap::new();

        #[derive(Default)]
        struct BuyOutcome {
            units: i64,
            weighted_profit: f64,
            weighted_cycle_hours: f64,
            latest: Option<DateTime<Utc>>,
        }

        let mut outcomes: HashMap<String, BuyOutcome> = HashMap::new();

        let mut sorted_transactions = transactions.to_vec();
        sorted_transactions.sort_by_key(|transaction| transaction.created_at);

        for transaction in sorted_transactions {
            if transaction.quantity <= 0 || transaction.price <= 0 {
                continue;
            }

            let key = item_key(&transaction.wfm_id, &transaction.sub_type);
            let unit_price = transaction.price as f64 / transaction.quantity as f64;

            if transaction.transaction_type == TransactionType::Purchase {
                lots_by_item.entry(key).or_default().push_back(PurchaseLot {
                    remaining: transaction.quantity,
                    unit_price,
                    at: transaction.created_at,
                    decision_id: transaction_property_string(&transaction, "algoframe_decision_id"),
                    predicted_profit: transaction_property_f64(
                        &transaction,
                        "algoframe_expected_profit",
                    ),
                });

                continue;
            }

            let Some(lots) = lots_by_item.get_mut(&key) else {
                continue;
            };

            let mut remaining = transaction.quantity;

            while remaining > 0 {
                let Some(mut lot) = lots.pop_front() else {
                    break;
                };

                let matched = remaining.min(lot.remaining);

                if let Some(decision_id) = &lot.decision_id {
                    let profit = unit_price - lot.unit_price;
                    let cycle_hours =
                        (transaction.created_at - lot.at).num_seconds().max(0) as f64 / 3600.0;

                    let outcome = outcomes.entry(decision_id.clone()).or_default();
                    outcome.units += matched;
                    outcome.weighted_profit += profit * matched as f64;
                    outcome.weighted_cycle_hours += cycle_hours * matched as f64;
                    outcome.latest = Some(transaction.created_at);
                }

                lot.remaining -= matched;
                remaining -= matched;

                if lot.remaining > 0 {
                    lots.push_front(lot);
                }
            }
        }

        for (decision_id, outcome) in outcomes {
            if outcome.units <= 0 {
                continue;
            }

            let Some(index) = decision_index.get(&decision_id).copied() else {
                continue;
            };

            let decision = &mut self.decisions[index];
            let actual_profit = outcome.weighted_profit / outcome.units as f64;
            let cycle_hours = outcome.weighted_cycle_hours / outcome.units as f64;

            decision.status = DecisionStatus::Completed;
            decision.lifecycle = LifecycleState::Sold;
            decision.completed_at = outcome.latest;
            decision.updated_at = outcome.latest.unwrap_or(now);
            decision.actual_profit = Some(actual_profit);
            decision.actual_cycle_hours = Some(cycle_hours);

            let reward = normalized_reward(
                actual_profit * outcome.units as f64,
                decision.capital.max(1.0),
                cycle_hours.max(0.05),
                self.config.human_minutes_per_trade,
                self.config.human_time_value_plat_per_hour,
            );

            decision.actual_reward = Some(reward);

            self.pending_outcomes.push(OutcomeRecord {
                id: format!("{}:cycle", decision.id),
                decision_id: decision.id.clone(),
                item_key: decision.item_key.clone(),
                side: TradeSide::Buy,
                outcome_type: "buy_to_sell_completed".into(),
                quantity: outcome.units,
                profit: actual_profit,
                reward,
                fill_hours: decision.actual_fill_hours.unwrap_or(0.0),
                cycle_hours,
                competition_response: decision.features.dynamics.competitive_response_score,
                simulated: false,
                created_at: outcome.latest.unwrap_or(now),
            });

            self.pending_decisions
                .insert(decision.id.clone(), decision.clone());
        }

        // Failure/censoring learning.
        let paper_config = self.config.clone();
        let paper_snapshots = self.snapshots.clone();

        for decision in &mut self.decisions {
            if !matches!(
                decision.status,
                DecisionStatus::Open | DecisionStatus::Partial | DecisionStatus::PaperOpen
            ) {
                continue;
            }

            if self.active_decision_ids.contains(&decision.id) {
                continue;
            }

            let age_hours = (now - decision.created_at).num_seconds().max(0) as f64 / 3600.0;

            if decision.status == DecisionStatus::PaperOpen {
                if age_hours >= decision.predicted_fill.expected_fill_hours.max(0.25) {
                    if let Some(outcome) =
                        resolve_paper_decision(&paper_config, &paper_snapshots, decision, age_hours)
                    {
                        self.pending_outcomes.push(outcome);
                        self.pending_decisions
                            .insert(decision.id.clone(), decision.clone());
                    }
                }
                continue;
            }

            if age_hours >= EXPIRED_ORDER_GRACE_HOURS {
                decision.status = DecisionStatus::Expired;
                decision.lifecycle = LifecycleState::Expired;
                decision.updated_at = now;
                decision.actual_cycle_hours = Some(age_hours);
                decision.actual_reward = Some(-0.02);

                self.pending_outcomes.push(OutcomeRecord {
                    id: format!("{}:expired", decision.id),
                    decision_id: decision.id.clone(),
                    item_key: decision.item_key.clone(),
                    side: decision.side,
                    outcome_type: "expired".into(),
                    quantity: decision.filled_quantity,
                    profit: 0.0,
                    reward: -0.02,
                    fill_hours: age_hours,
                    cycle_hours: age_hours,
                    competition_response: decision.features.dynamics.competitive_response_score,
                    simulated: false,
                    created_at: now,
                });

                self.pending_decisions
                    .insert(decision.id.clone(), decision.clone());
            }
        }

        // Learn from non-actions as well. A rejected BUY is replayed against
        // the recorded external market after enough time has elapsed. If a
        // plausible flip later existed, "skip" receives regret; otherwise it
        // receives a small positive reward for preserving capital.
        let skip_snapshots = self.snapshots.clone();
        let skip_config = self.config.clone();

        for decision in &mut self.decisions {
            if decision.status != DecisionStatus::Rejected
                || decision.actual_reward.is_some()
                || decision.side != TradeSide::Buy
                || !decision.explanation.guardrails.is_empty()
            {
                continue;
            }

            let age_hours = (now - decision.created_at).num_seconds().max(0) as f64 / 3600.0;

            if age_hours < 6.0 {
                continue;
            }

            if let Some(outcome) = resolve_skipped_decision(
                &skip_config,
                &skip_snapshots,
                decision,
                age_hours.min(24.0),
            ) {
                self.pending_outcomes.push(outcome);
                self.pending_decisions
                    .insert(decision.id.clone(), decision.clone());
            }
        }
    }

    fn rebuild_learning_indexes(&mut self) {
        self.bandit.clear();
        self.embedding_history.clear();

        for decision in &self.decisions {
            let key = (
                decision.side,
                decision.context_key.clone(),
                decision.chosen_action.clone(),
            );

            let evidence = self.bandit.entry(key).or_default();

            let weight = recency_weight(decision.updated_at, self.config.recency_half_life_days);

            match decision.status {
                DecisionStatus::Completed
                | DecisionStatus::Filled
                | DecisionStatus::PaperCompleted => {
                    evidence.successes += weight;
                    evidence.trials += weight;
                }
                DecisionStatus::Expired | DecisionStatus::Cancelled => {
                    evidence.failures += weight;
                    evidence.trials += weight;
                }
                _ => {}
            }

            if let Some(reward) = decision.actual_reward {
                evidence.reward_samples.push(WeightedSample {
                    value: reward,
                    weight,
                });

                self.embedding_history
                    .push((decision.features.embedding.clone(), reward, weight));
            }

            let exposure = decision
                .actual_fill_hours
                .or(decision.actual_cycle_hours)
                .unwrap_or_else(|| {
                    (Utc::now() - decision.created_at).num_seconds().max(0) as f64 / 3600.0
                })
                .max(0.05);

            evidence.fill_observations.push(SurvivalObservation {
                hours: exposure,
                event: matches!(
                    decision.status,
                    DecisionStatus::Completed
                        | DecisionStatus::Filled
                        | DecisionStatus::PaperCompleted
                ),
                weight,
            });
        }

        self.embedding_feature_weights = learned_embedding_weights(&self.decisions);
    }

    fn queue_alert(
        &mut self,
        severity: &str,
        code: &str,
        message: String,
        item_key: Option<String>,
    ) {
        self.pending_alerts.push(AlertRecord {
            id: Uuid::new_v4().to_string(),
            severity: severity.to_string(),
            code: code.to_string(),
            message,
            item_key,
            created_at: Utc::now(),
            acknowledged: false,
        });
    }

    fn auto_tune(&mut self) {
        if !self.config.automatic_tuning || self.decisions.len() < 100 {
            return;
        }

        let completed: Vec<&DecisionRecord> = self
            .decisions
            .iter()
            .rev()
            .filter(|decision| decision.actual_reward.is_some())
            .take(200)
            .collect();

        if completed.len() < 50 {
            return;
        }

        let prediction_errors: Vec<f64> = completed
            .iter()
            .filter_map(|decision| {
                decision
                    .actual_profit
                    .map(|actual| (actual - decision.predicted_profit).abs())
            })
            .collect();

        let mae = mean(&prediction_errors);
        let typical_profit = mean(
            &completed
                .iter()
                .filter_map(|decision| decision.actual_profit)
                .map(f64::abs)
                .collect::<Vec<_>>(),
        )
        .max(1.0);

        let normalized_error = mae / typical_profit;

        let previous = self.config.recency_half_life_days;

        if normalized_error > 0.60 {
            self.config.recency_half_life_days =
                (self.config.recency_half_life_days * 0.90).max(7.0);
            self.config.challenger_fraction = (self.config.challenger_fraction * 0.85).max(0.02);
        } else if normalized_error < 0.25 && self.health.healthy {
            self.config.recency_half_life_days =
                (self.config.recency_half_life_days * 1.03).min(60.0);
            self.config.challenger_fraction = (self.config.challenger_fraction * 1.02).min(0.15);
        }

        if (previous - self.config.recency_half_life_days).abs() > f64::EPSILON {
            self.config.settings_revision = self.config.settings_revision.saturating_add(1);
            self.config_dirty = true;
        }
    }

    async fn run_champion_challenger(&mut self, db: &DatabaseConnection) -> Result<(), Error> {
        let champion_shadow = format!("shadow_{}", self.config.champion_policy);
        let challenger_shadow = format!("shadow_{}", self.config.challenger_policy);

        let champion = offline_policy_evaluation(&self.decisions, &champion_shadow);
        let challenger = offline_policy_evaluation(&self.decisions, &challenger_shadow);

        LearningStore::insert_evaluation(
            db,
            &format!("champion:{}", self.config.champion_policy),
            &champion,
        )
        .await?;
        LearningStore::insert_evaluation(
            db,
            &format!("challenger:{}", self.config.challenger_policy),
            &challenger,
        )
        .await?;

        let champion_model = ModelVersionRecord {
            id: format!("policy:{}", self.config.champion_policy),
            version: format!("{}:{}", MODEL_VERSION, self.config.champion_policy),
            role: ModelRole::Champion,
            active: !self.health.fallback_active,
            parameters: serde_json::json!({
                "policy": self.config.champion_policy,
                "settings_revision": self.config.settings_revision,
                "recency_half_life_days": self.config.recency_half_life_days,
            }),
            trained_on_decisions: self.decisions.len(),
            average_reward: champion.metrics.average_reward,
            failure_rate: champion.metrics.failure_rate,
            prediction_mae: self.health.recent_prediction_mae,
            calibration_error: self.health.calibration_error,
            drawdown: self.health.drawdown,
            created_at: Utc::now(),
            promoted_at: None,
        };

        let challenger_model = ModelVersionRecord {
            id: format!("policy:{}", self.config.challenger_policy),
            version: format!("{}:{}", MODEL_VERSION, self.config.challenger_policy),
            role: ModelRole::Challenger,
            active: false,
            parameters: serde_json::json!({
                "policy": self.config.challenger_policy,
                "settings_revision": self.config.settings_revision,
                "recency_half_life_days": self.config.recency_half_life_days,
            }),
            trained_on_decisions: self.decisions.len(),
            average_reward: challenger.metrics.average_reward,
            failure_rate: challenger.metrics.failure_rate,
            prediction_mae: self.health.recent_prediction_mae,
            calibration_error: self.health.calibration_error,
            drawdown: self.health.drawdown,
            created_at: Utc::now(),
            promoted_at: None,
        };

        LearningStore::upsert_model(db, &champion_model).await?;
        LearningStore::upsert_model(db, &challenger_model).await?;

        if !self.config.automatic_promotion
            || challenger.metrics.sample_count < self.config.minimum_promotion_samples
        {
            return Ok(());
        }

        let champion_reward = champion
            .metrics
            .doubly_robust_reward
            .max(champion.metrics.average_reward);
        let challenger_reward = challenger
            .metrics
            .doubly_robust_reward
            .max(challenger.metrics.average_reward);

        let reward_threshold =
            champion_reward + champion_reward.abs() * self.config.promotion_margin_pct;

        let failure_ok = challenger.metrics.failure_rate
            <= champion.metrics.failure_rate + self.config.maximum_challenger_failure_delta;

        // Promotion requires a non-overlapping lower confidence bound, not just
        // a better point estimate.
        let confidence_ok = challenger.confidence_low > champion.confidence_high;

        let drawdown_ok = challenger.metrics.max_drawdown
            <= champion
                .metrics
                .max_drawdown
                .max(self.config.max_drawdown_pct);

        let category_coverage: HashSet<String> = self
            .decisions
            .iter()
            .filter(|decision| decision.actual_reward.is_some())
            .map(|decision| decision.category.clone())
            .collect();

        let coverage_ok = category_coverage.len() >= 3;

        if challenger_reward > reward_threshold
            && failure_ok
            && confidence_ok
            && drawdown_ok
            && coverage_ok
        {
            let previous_champion = self.config.champion_policy.clone();
            self.config.champion_policy = self.config.challenger_policy.clone();
            self.config.challenger_policy = previous_champion;
            self.config.challenger_fraction = 0.10;
            self.config.settings_revision = self.config.settings_revision.saturating_add(1);
            self.config_dirty = true;

            let promoted = ModelVersionRecord {
                role: ModelRole::Champion,
                active: true,
                promoted_at: Some(Utc::now()),
                ..challenger_model
            };
            let demoted = ModelVersionRecord {
                role: ModelRole::Challenger,
                active: false,
                ..champion_model
            };

            LearningStore::upsert_model(db, &promoted).await?;
            LearningStore::upsert_model(db, &demoted).await?;

            self.queue_alert(
                "info",
                "challenger_promoted",
                format!(
                    "Policy '{}' promoted over '{}' ({:.4} vs {:.4} DR reward).",
                    self.config.champion_policy,
                    self.config.challenger_policy,
                    challenger_reward,
                    champion_reward
                ),
                None,
            );
        }

        Ok(())
    }
}

fn resolve_skipped_decision(
    config: &UltimateConfig,
    snapshots: &[MarketSnapshot],
    decision: &mut DecisionRecord,
    horizon_hours: f64,
) -> Option<OutcomeRecord> {
    let horizon_end =
        decision.created_at + chrono::Duration::seconds((horizon_hours * 3600.0) as i64);

    let mut relevant: Vec<&MarketSnapshot> = snapshots
        .iter()
        .filter(|snapshot| {
            snapshot.item_key == decision.item_key
                && snapshot.created_at > decision.created_at
                && snapshot.created_at <= horizon_end
        })
        .collect();

    relevant.sort_by_key(|snapshot| snapshot.created_at);

    if relevant.is_empty() {
        return None;
    }

    let hypothetical_buy = decision.features.robust_bid.max(1).saturating_add(1);

    let fill = relevant.iter().find(|snapshot| {
        snapshot.features.robust_ask > 0 && snapshot.features.robust_ask <= hypothetical_buy
    });

    let (missed_profit, cycle_hours) = if let Some(fill_snapshot) = fill {
        let best_future_bid = relevant
            .iter()
            .filter(|snapshot| snapshot.created_at >= fill_snapshot.created_at)
            .map(|snapshot| snapshot.features.robust_bid)
            .max()
            .unwrap_or(0);

        let profit = (best_future_bid - hypothetical_buy).max(0) as f64;

        let cycle = relevant
            .iter()
            .filter(|snapshot| snapshot.created_at >= fill_snapshot.created_at)
            .find(|snapshot| snapshot.features.robust_bid >= hypothetical_buy.saturating_add(1))
            .map(|snapshot| {
                (snapshot.created_at - decision.created_at)
                    .num_seconds()
                    .max(0) as f64
                    / 3600.0
            })
            .unwrap_or(horizon_hours);

        (profit, cycle.max(0.05))
    } else {
        (0.0, horizon_hours.max(0.05))
    };

    let reward = if missed_profit > 0.0 {
        -normalized_reward(
            missed_profit,
            hypothetical_buy as f64,
            cycle_hours,
            config.human_minutes_per_trade,
            config.human_time_value_plat_per_hour,
        )
        .abs()
        .min(5.0)
    } else {
        // Small positive value for correctly preserving capital. Keep this
        // intentionally tiny so the model does not learn pathological inertia.
        0.01
    };

    decision.chosen_action = "skip".to_string();
    decision.chosen_propensity = 1.0;
    decision.propensities = HashMap::from([("skip".to_string(), 1.0)]);
    decision.actual_profit = Some(-missed_profit);
    decision.actual_reward = Some(reward);
    decision.actual_cycle_hours = Some(cycle_hours);
    decision.updated_at = Utc::now();

    Some(OutcomeRecord {
        id: format!("{}:skip-counterfactual", decision.id),
        decision_id: decision.id.clone(),
        item_key: decision.item_key.clone(),
        side: decision.side,
        outcome_type: "skip_counterfactual".into(),
        quantity: 0,
        profit: -missed_profit,
        reward,
        fill_hours: 0.0,
        cycle_hours,
        competition_response: decision.features.dynamics.competitive_response_score,
        simulated: true,
        created_at: Utc::now(),
    })
}

fn resolve_paper_decision(
    config: &UltimateConfig,
    snapshots: &[MarketSnapshot],
    decision: &mut DecisionRecord,
    age_hours: f64,
) -> Option<OutcomeRecord> {
    let relevant: Vec<&MarketSnapshot> = snapshots
        .iter()
        .filter(|snapshot| {
            snapshot.item_key == decision.item_key && snapshot.created_at >= decision.created_at
        })
        .collect();

    if relevant.is_empty() {
        return None;
    }

    let crossed = match decision.side {
        TradeSide::Buy => relevant.iter().any(|snapshot| {
            snapshot.features.robust_ask > 0 && snapshot.features.robust_ask <= decision.price
        }),
        TradeSide::Sell => relevant
            .iter()
            .any(|snapshot| snapshot.features.robust_bid >= decision.price),
    };

    let probabilistic_fill = stable_fraction(&[&decision.id, &format!("{:.2}", age_hours)])
        <= decision.predicted_fill.fill_24h;

    if !crossed && !probabilistic_fill {
        return None;
    }

    decision.status = DecisionStatus::PaperCompleted;
    decision.lifecycle = LifecycleState::PaperSimulated;
    decision.filled_quantity = decision.quantity;
    decision.fill_ratio = 1.0;
    decision.actual_fill_hours = Some(age_hours);
    decision.actual_cycle_hours = Some(age_hours);

    let simulated_profit = decision.predicted_profit
        * (0.85 + stable_fraction(&[&decision.id, "profit_noise"]) * 0.30);

    let reward = normalized_reward(
        simulated_profit * decision.quantity as f64,
        decision.capital.max(1.0),
        age_hours.max(0.05),
        config.human_minutes_per_trade,
        config.human_time_value_plat_per_hour,
    );

    decision.actual_profit = Some(simulated_profit);
    decision.actual_reward = Some(reward);
    decision.completed_at = Some(Utc::now());
    decision.updated_at = Utc::now();

    Some(OutcomeRecord {
        id: format!("{}:paper", decision.id),
        decision_id: decision.id.clone(),
        item_key: decision.item_key.clone(),
        side: decision.side,
        outcome_type: "paper_simulated".into(),
        quantity: decision.quantity,
        profit: simulated_profit,
        reward,
        fill_hours: age_hours,
        cycle_hours: age_hours,
        competition_response: decision.features.dynamics.competitive_response_score,
        simulated: true,
        created_at: Utc::now(),
    })
}

fn build_learned_state(transactions: &[TransactionModel], config: &UltimateConfig) -> LearnedState {
    let mut grouped: HashMap<String, Vec<TransactionModel>> = HashMap::new();

    for transaction in transactions {
        grouped
            .entry(item_key(&transaction.wfm_id, &transaction.sub_type))
            .or_default()
            .push(transaction.clone());
    }

    let mut item_profiles = HashMap::new();
    let mut category_accumulators: HashMap<String, ProfileAccumulator> = HashMap::new();
    let mut global_accumulator = ProfileAccumulator::default();

    for (key, mut item_transactions) in grouped {
        item_transactions.sort_by_key(|transaction| transaction.created_at);

        let accumulator = ProfileAccumulator::from_transactions(&item_transactions, config);

        global_accumulator.merge(&accumulator);

        let category = infer_transaction_category(&item_transactions);

        category_accumulators
            .entry(category)
            .or_default()
            .merge(&accumulator);

        item_profiles.insert(key, accumulator.profile());
    }

    let category_profiles = category_accumulators
        .into_iter()
        .map(|(category, accumulator)| (category, accumulator.profile()))
        .collect();

    LearnedState {
        item_profiles,
        category_profiles,
        global: global_accumulator.profile(),
        transaction_count: transactions.len(),
    }
}

impl ProfileAccumulator {
    fn from_transactions(transactions: &[TransactionModel], config: &UltimateConfig) -> Self {
        let mut result = Self::default();
        let mut lots: VecDeque<PurchaseLot> = VecDeque::new();

        for transaction in transactions {
            if transaction.quantity <= 0 || transaction.price <= 0 {
                continue;
            }

            result.trade_count += 1;

            let quantity = transaction.quantity;
            let unit_price = transaction.price as f64 / quantity as f64;

            let weight = recency_weight(transaction.created_at, config.recency_half_life_days)
                * quantity as f64;

            if transaction.transaction_type == TransactionType::Purchase {
                result.purchase_units += quantity;

                if let Some(hours) = transaction_order_age_hours(transaction) {
                    result.buy_fill_hours.push(WeightedSample {
                        value: hours,
                        weight,
                    });
                }

                lots.push_back(PurchaseLot {
                    remaining: quantity,
                    unit_price,
                    at: transaction.created_at,
                    decision_id: transaction_property_string(transaction, "algoframe_decision_id"),
                    predicted_profit: transaction_property_f64(
                        transaction,
                        "algoframe_expected_profit",
                    ),
                });

                continue;
            }

            result.sold_units += quantity;
            result.sale_prices.push(WeightedSample {
                value: unit_price,
                weight,
            });

            if let Some(hours) = transaction_order_age_hours(transaction) {
                result.sell_fill_hours.push(WeightedSample {
                    value: hours,
                    weight,
                });
            }

            let mut remaining = quantity;

            while remaining > 0 {
                let Some(mut lot) = lots.pop_front() else {
                    break;
                };

                let matched = remaining.min(lot.remaining);
                let profit = unit_price - lot.unit_price;
                let roi = if lot.unit_price > 0.0 {
                    profit / lot.unit_price
                } else {
                    0.0
                };

                let hold_hours =
                    (transaction.created_at - lot.at).num_seconds().max(0) as f64 / 3600.0;

                let matched_weight =
                    recency_weight(transaction.created_at, config.recency_half_life_days)
                        * matched as f64;

                result.matched_units += matched;

                result.profits.push(WeightedSample {
                    value: profit,
                    weight: matched_weight,
                });
                result.rois.push(WeightedSample {
                    value: roi,
                    weight: matched_weight,
                });
                result.hold_hours.push(WeightedSample {
                    value: hold_hours,
                    weight: matched_weight,
                });

                if let Some(predicted) = lot.predicted_profit {
                    let error = profit - predicted;

                    result.prediction_errors.push(WeightedSample {
                        value: error,
                        weight: matched_weight,
                    });
                    result.prediction_abs_errors.push(WeightedSample {
                        value: error.abs(),
                        weight: matched_weight,
                    });
                }

                lot.remaining -= matched;
                remaining -= matched;

                if lot.remaining > 0 {
                    lots.push_front(lot);
                }
            }
        }

        for lot in lots {
            if lot.remaining <= 0 {
                continue;
            }

            result.open_units += lot.remaining;

            let age_hours = (Utc::now() - lot.at).num_seconds().max(0) as f64 / 3600.0;

            result.open_age_hours.push(WeightedSample {
                value: age_hours,
                weight: lot.remaining as f64,
            });
        }

        result
    }

    fn merge(&mut self, other: &Self) {
        self.purchase_units += other.purchase_units;
        self.sold_units += other.sold_units;
        self.matched_units += other.matched_units;
        self.open_units += other.open_units;
        self.trade_count += other.trade_count;

        self.profits.extend_from_slice(&other.profits);
        self.rois.extend_from_slice(&other.rois);
        self.hold_hours.extend_from_slice(&other.hold_hours);
        self.sale_prices.extend_from_slice(&other.sale_prices);
        self.buy_fill_hours.extend_from_slice(&other.buy_fill_hours);
        self.sell_fill_hours
            .extend_from_slice(&other.sell_fill_hours);
        self.open_age_hours.extend_from_slice(&other.open_age_hours);
        self.prediction_errors
            .extend_from_slice(&other.prediction_errors);
        self.prediction_abs_errors
            .extend_from_slice(&other.prediction_abs_errors);
    }

    fn profile(&self) -> ItemProfile {
        let avg_profit = robust_weighted_mean(&self.profits, 0.0);
        let avg_roi = robust_weighted_mean(&self.rois, 0.0);
        let avg_hold_hours = robust_weighted_mean(&self.hold_hours, 0.0);
        let avg_sale_price = robust_weighted_mean(&self.sale_prices, 0.0);
        let avg_buy_fill_hours = robust_weighted_mean(&self.buy_fill_hours, 0.0);
        let avg_sell_fill_hours = robust_weighted_mean(&self.sell_fill_hours, 0.0);
        let avg_open_age_hours = robust_weighted_mean(&self.open_age_hours, 0.0);

        let prediction_bias = robust_weighted_mean(&self.prediction_errors, 0.0);
        let prediction_mae = robust_weighted_mean(&self.prediction_abs_errors, 0.0);

        let prediction_accuracy = if self.prediction_abs_errors.is_empty() {
            0.50
        } else {
            (1.0 / (1.0 + prediction_mae / (avg_profit.abs() + 5.0))).clamp(0.10, 1.0)
        };

        let profit_mad = weighted_mad(&self.profits).unwrap_or(0.0);
        let profit_stability = if self.profits.is_empty() {
            0.50
        } else {
            (1.0 / (1.0 + profit_mad / (avg_profit.abs() + 5.0))).clamp(0.15, 1.0)
        };

        let sample_weight = total_weight(&self.profits);
        let sample_confidence = sample_weight / (sample_weight + 14.0);

        let trade_confidence = self.trade_count as f64 / (self.trade_count as f64 + 8.0);

        let confidence = (sample_confidence * 0.85 + trade_confidence * 0.15).clamp(0.0, 1.0);

        let sell_through = if self.purchase_units > 0 {
            (self.matched_units as f64 / self.purchase_units as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        ItemProfile {
            purchase_units: self.purchase_units,
            sold_units: self.sold_units,
            matched_units: self.matched_units,
            open_units: self.open_units,
            trade_count: self.trade_count,
            avg_profit,
            avg_roi,
            avg_hold_hours,
            avg_sale_price,
            avg_buy_fill_hours,
            avg_sell_fill_hours,
            avg_open_age_hours,
            prediction_bias,
            prediction_mae,
            prediction_accuracy,
            sell_through,
            profit_stability,
            confidence,
        }
    }
}

fn item_key(wfm_id: &str, sub_type: &Option<SubType>) -> String {
    let subtype = serde_json::to_string(sub_type).unwrap_or_else(|_| "null".to_string());

    format!("{wfm_id}|{subtype}")
}

fn historical_center(price: &ItemPriceInfo) -> f64 {
    if price.median > 0.0 {
        price.median
    } else if price.moving_avg.unwrap_or(0.0) > 0.0 {
        price.moving_avg.unwrap_or(0.0)
    } else {
        price.avg_price.max(0.0)
    }
}

fn price_volatility(price: &ItemPriceInfo) -> f64 {
    let denominator = historical_center(price).max(1.0);

    ((price.max_price - price.min_price).abs() / denominator).clamp(0.0, 4.0)
}

fn liquidity_score(volume: f64) -> f64 {
    (1.0 - (-volume.max(0.0) / 20.0).exp()).clamp(0.0, 1.0)
}

fn inventory_pressure(profile: &ItemProfile) -> f64 {
    if profile.purchase_units <= 0 || profile.open_units <= 0 {
        return 0.0;
    }

    let open_ratio = profile.open_units as f64 / profile.purchase_units.max(1) as f64;

    let age_multiplier = 1.0 + (profile.avg_open_age_hours / 168.0).clamp(0.0, 2.0);

    (open_ratio * age_multiplier).clamp(0.0, 1.0)
}

fn context_key(
    side: TradeSide,
    category: &str,
    family: &str,
    item_key: &str,
    features: &MarketFeatures,
) -> String {
    format!(
        "{}|{}|family:{}|item:{}|liq:{}|spr:{}|vol:{}|inv:{}|chg:{}|reg:{}|life:{}|hour:{}|day:{}",
        side_name(side),
        category,
        family,
        item_key,
        bucket(features.liquidity, &[0.20, 0.45, 0.70]),
        bucket(features.spread_percent.abs(), &[0.08, 0.18, 0.35]),
        bucket(features.volatility, &[0.15, 0.35, 0.70]),
        bucket(features.inventory_pressure, &[0.20, 0.45, 0.70]),
        bucket(features.dynamics.churn_score, &[0.20, 0.50, 0.75]),
        features.regime.as_str(),
        features.lifecycle.as_str(),
        bucket(features.time_hour_sin, &[-0.5, 0.0, 0.5]),
        bucket(features.weekday_sin, &[-0.5, 0.0, 0.5]),
    )
}

fn side_name(side: TradeSide) -> &'static str {
    match side {
        TradeSide::Buy => "buy",
        TradeSide::Sell => "sell",
    }
}

fn bucket(value: f64, cuts: &[f64]) -> usize {
    cuts.iter()
        .position(|cut| value < *cut)
        .unwrap_or(cuts.len())
}

fn action_prior(side: TradeSide, price_offset: i64) -> (f64, f64) {
    match side {
        TradeSide::Buy => {
            if price_offset <= 0 {
                (2.7, 2.3)
            } else if price_offset == 1 {
                (3.0, 2.0)
            } else {
                (2.6, 2.4)
            }
        }
        TradeSide::Sell => {
            if price_offset <= 0 {
                (2.4, 2.6)
            } else if price_offset == 1 {
                (3.0, 2.0)
            } else {
                (3.1, 1.9)
            }
        }
    }
}

fn exponential_survival(hours: f64) -> SurvivalPrediction {
    let hours = hours.max(0.10);

    let fill = |h: f64| (1.0 - (-h / hours).exp()).clamp(0.01, 0.99);

    SurvivalPrediction {
        fill_15m: fill(0.25),
        fill_1h: fill(1.0),
        fill_6h: fill(6.0),
        fill_24h: fill(24.0),
        median_fill_hours: hours * std::f64::consts::LN_2,
        expected_fill_hours: hours,
        calibration: 0.5,
    }
}

fn merge_evidence(
    exact: &BanditEvidence,
    category: &BanditEvidence,
    global: &BanditEvidence,
) -> BanditEvidence {
    let mut merged = BanditEvidence {
        successes: exact.successes + category.successes * 0.35 + global.successes * 0.12,
        failures: exact.failures + category.failures * 0.35 + global.failures * 0.12,
        trials: exact.trials + category.trials * 0.35 + global.trials * 0.12,
        reward_samples: exact.reward_samples.clone(),
        fill_observations: exact.fill_observations.clone(),
    };

    merged
        .reward_samples
        .extend(category.reward_samples.iter().map(|sample| WeightedSample {
            value: sample.value,
            weight: sample.weight * 0.35,
        }));

    merged
        .reward_samples
        .extend(global.reward_samples.iter().map(|sample| WeightedSample {
            value: sample.value,
            weight: sample.weight * 0.12,
        }));

    merged
        .fill_observations
        .extend(
            category
                .fill_observations
                .iter()
                .map(|observation| SurvivalObservation {
                    hours: observation.hours,
                    event: observation.event,
                    weight: observation.weight * 0.35,
                }),
        );

    merged
        .fill_observations
        .extend(
            global
                .fill_observations
                .iter()
                .map(|observation| SurvivalObservation {
                    hours: observation.hours,
                    event: observation.event,
                    weight: observation.weight * 0.12,
                }),
        );

    merged
}

fn merge_evidence_many(evidence: &[BanditEvidence], weight: f64) -> BanditEvidence {
    let mut merged = BanditEvidence::default();

    for item in evidence {
        merged.successes += item.successes * weight;
        merged.failures += item.failures * weight;
        merged.trials += item.trials * weight;

        merged
            .reward_samples
            .extend(item.reward_samples.iter().map(|sample| WeightedSample {
                value: sample.value,
                weight: sample.weight * weight,
            }));

        merged
            .fill_observations
            .extend(
                item.fill_observations
                    .iter()
                    .map(|observation| SurvivalObservation {
                        hours: observation.hours,
                        event: observation.event,
                        weight: observation.weight * weight,
                    }),
            );
    }

    merged
}

fn apply_policy_utility(
    policy: &str,
    side: TradeSide,
    candidate: &CandidateAction,
    base: f64,
) -> f64 {
    match policy {
        "conservative" => {
            base * (1.0 - candidate.uncertainty * 0.45).clamp(0.25, 1.0)
                * (0.70 + candidate.survival.fill_6h * 0.30)
        }
        "aggressive" => {
            let price_push = match side {
                TradeSide::Buy => 1.0 + candidate.price_offset.max(0) as f64 * 0.025,
                TradeSide::Sell => 1.0 + candidate.price_offset.max(0) as f64 * 0.035,
            };

            base * price_push * (1.0 + candidate.uncertainty * 0.08)
        }
        "fast_turnover" => {
            base * (1.0 + (1.0 / candidate.survival.expected_fill_hours.max(0.25)).min(3.0) * 0.20)
        }
        "max_reward" => base * 0.35 + candidate.reward.final_reward * 0.65,
        _ => base,
    }
}

fn best_candidate_for_policy<'a>(
    policy: &str,
    side: TradeSide,
    candidates: &'a [CandidateAction],
) -> &'a CandidateAction {
    match policy {
        "conservative" => candidates
            .iter()
            .max_by(|a, b| {
                let ua = apply_policy_utility(policy, side, a, a.reward.final_reward);
                let ub = apply_policy_utility(policy, side, b, b.reward.final_reward);
                ua.partial_cmp(&ub).unwrap_or(Ordering::Equal)
            })
            .unwrap_or_else(|| conservative_candidate(side, candidates)),
        "aggressive" => candidates
            .iter()
            .max_by(|a, b| {
                let ua = apply_policy_utility(policy, side, a, a.reward.final_reward);
                let ub = apply_policy_utility(policy, side, b, b.reward.final_reward);
                ua.partial_cmp(&ub).unwrap_or(Ordering::Equal)
            })
            .unwrap_or_else(|| aggressive_candidate(side, candidates)),
        "fast_turnover" => fastest_candidate(candidates),
        "max_reward" => max_reward_candidate(candidates),
        _ => max_reward_candidate(candidates),
    }
}

fn conservative_candidate<'a>(
    side: TradeSide,
    candidates: &'a [CandidateAction],
) -> &'a CandidateAction {
    match side {
        TradeSide::Buy => candidates
            .iter()
            .min_by(|a, b| a.price.cmp(&b.price).then(a.quantity.cmp(&b.quantity)))
            .unwrap(),
        TradeSide::Sell => candidates
            .iter()
            .max_by(|a, b| a.price.cmp(&b.price).then(b.quantity.cmp(&a.quantity)))
            .unwrap(),
    }
}

fn aggressive_candidate<'a>(
    side: TradeSide,
    candidates: &'a [CandidateAction],
) -> &'a CandidateAction {
    match side {
        TradeSide::Buy => candidates
            .iter()
            .max_by(|a, b| a.price.cmp(&b.price).then(a.quantity.cmp(&b.quantity)))
            .unwrap(),
        TradeSide::Sell => candidates
            .iter()
            .min_by(|a, b| a.price.cmp(&b.price).then(b.quantity.cmp(&a.quantity)))
            .unwrap(),
    }
}

fn fastest_candidate(candidates: &[CandidateAction]) -> &CandidateAction {
    candidates
        .iter()
        .min_by(|a, b| {
            a.survival
                .expected_fill_hours
                .partial_cmp(&b.survival.expected_fill_hours)
                .unwrap_or(Ordering::Equal)
        })
        .unwrap()
}

fn max_reward_candidate(candidates: &[CandidateAction]) -> &CandidateAction {
    candidates
        .iter()
        .max_by(|a, b| {
            a.reward
                .final_reward
                .partial_cmp(&b.reward.final_reward)
                .unwrap_or(Ordering::Equal)
        })
        .unwrap()
}

fn blend_profile(base: &ItemProfile, newer: &ItemProfile, weight: f64) -> ItemProfile {
    let weight = weight.clamp(0.0, 1.0);
    let inverse = 1.0 - weight;
    let mix = |a: f64, b: f64| a * inverse + b * weight;

    ItemProfile {
        purchase_units: newer.purchase_units,
        sold_units: newer.sold_units,
        matched_units: newer.matched_units,
        open_units: newer.open_units,
        trade_count: newer.trade_count,
        avg_profit: mix(base.avg_profit, newer.avg_profit),
        avg_roi: mix(base.avg_roi, newer.avg_roi),
        avg_hold_hours: mix(base.avg_hold_hours, newer.avg_hold_hours),
        avg_sale_price: mix(base.avg_sale_price, newer.avg_sale_price),
        avg_buy_fill_hours: mix(base.avg_buy_fill_hours, newer.avg_buy_fill_hours),
        avg_sell_fill_hours: mix(base.avg_sell_fill_hours, newer.avg_sell_fill_hours),
        avg_open_age_hours: newer.avg_open_age_hours,
        prediction_bias: mix(base.prediction_bias, newer.prediction_bias),
        prediction_mae: mix(base.prediction_mae, newer.prediction_mae),
        prediction_accuracy: mix(base.prediction_accuracy, newer.prediction_accuracy),
        sell_through: mix(base.sell_through, newer.sell_through),
        profit_stability: mix(base.profit_stability, newer.profit_stability),
        confidence: newer.confidence,
    }
}

fn transaction_property_f64(transaction: &TransactionModel, key: &str) -> Option<f64> {
    let value = transaction.properties.as_ref()?.get(key)?;

    value
        .as_f64()
        .or_else(|| value.as_i64().map(|value| value as f64))
        .or_else(|| value.as_u64().map(|value| value as f64))
}

fn transaction_property_string(transaction: &TransactionModel, key: &str) -> Option<String> {
    transaction
        .properties
        .as_ref()?
        .get(key)?
        .as_str()
        .map(str::to_string)
}

fn transaction_order_age_hours(transaction: &TransactionModel) -> Option<f64> {
    let started = transaction_property_string(transaction, "algoframe_order_started_at")?;

    let started = DateTime::parse_from_rfc3339(&started)
        .ok()?
        .with_timezone(&Utc);

    Some((transaction.created_at - started).num_seconds().max(0) as f64 / 3600.0)
}

fn infer_transaction_category(transactions: &[TransactionModel]) -> String {
    let tags: Vec<String> = transactions
        .iter()
        .flat_map(|transaction| {
            transaction
                .tags
                .split(',')
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect();

    let item_name = transactions
        .first()
        .map(|transaction| transaction.item_name.as_str())
        .unwrap_or("");

    category_from(&tags, item_name)
}

fn profit_attribution(completed: &[&DecisionRecord]) -> ProfitAttribution {
    let mut result = ProfitAttribution::default();

    for decision in completed {
        let actual = decision.actual_profit.unwrap_or(0.0);

        result.total_realized += actual;

        result.market_selection += (actual.min(decision.predicted_profit).max(0.0)) * 0.45;

        result.buy_price_optimization += if decision.side == TradeSide::Buy {
            (decision.features.robust_ask as f64 - decision.price as f64)
                .max(0.0)
                .min(actual.max(0.0))
                * 0.20
        } else {
            0.0
        };

        result.sell_price_optimization += if decision.side == TradeSide::Sell {
            (decision.price as f64 - decision.features.robust_bid as f64)
                .max(0.0)
                .min(actual.max(0.0))
                * 0.20
        } else {
            0.0
        };

        result.arbitrage += decision.reward_breakdown.arbitrage_bonus;
        result.event_intelligence += decision.reward_breakdown.event_adjustment;
        result.human_time_cost -= decision.reward_breakdown.human_time_cost;

        if decision.model_role == ModelRole::Challenger && actual < 0.0 {
            result.exploration_cost += actual;
        }
    }

    result.portfolio_allocation = result.total_realized
        - result.market_selection
        - result.buy_price_optimization
        - result.sell_price_optimization
        - result.arbitrage
        - result.event_intelligence
        - result.exploration_cost
        - result.human_time_cost;

    result
}

const MAX_PREDICTION_BIAS_WEIGHT: f64 = 0.55;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_key_changes_with_regime() {
        let mut features = MarketFeatures::default();
        features.regime = MarketRegime::Normal;

        let a = context_key(
            TradeSide::Buy,
            "prime",
            "xaku prime",
            "xaku|none",
            &features,
        );

        features.regime = MarketRegime::Shock;
        let b = context_key(
            TradeSide::Buy,
            "prime",
            "xaku prime",
            "xaku|none",
            &features,
        );

        assert_ne!(a, b);
    }

    #[test]
    fn fallback_candidate_is_conservative() {
        let candidates = vec![
            CandidateAction {
                key: "a".into(),
                side: TradeSide::Buy,
                price: 10,
                quantity: 1,
                price_offset: 0,
                expected_sell_price: 20.0,
                survival: SurvivalPrediction::default(),
                reward: RewardBreakdown::default(),
                bandit_sample: 0.0,
                propensity: 0.0,
                uncertainty: 0.0,
                valid: true,
                reject_reasons: vec![],
            },
            CandidateAction {
                key: "b".into(),
                side: TradeSide::Buy,
                price: 12,
                quantity: 2,
                price_offset: 2,
                expected_sell_price: 20.0,
                survival: SurvivalPrediction::default(),
                reward: RewardBreakdown::default(),
                bandit_sample: 0.0,
                propensity: 0.0,
                uncertainty: 0.0,
                valid: true,
                reject_reasons: vec![],
            },
        ];

        assert_eq!(
            conservative_candidate(TradeSide::Buy, &candidates).price,
            10
        );
    }
}
