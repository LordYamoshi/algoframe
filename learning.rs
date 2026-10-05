use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    fs,
    path::PathBuf,
};

use chrono::{DateTime, Utc};
use entity::{
    enums::TransactionType,
    transaction::{Model as TransactionModel, TransactionPaginationQueryDto},
};
use rand::thread_rng;
use rand_distr::{Beta, Distribution, Normal};
use serde::{Deserialize, Serialize};
use service::{sea_orm::DatabaseConnection, TransactionQuery};
use utils::{get_location, Error, SubType};
use uuid::Uuid;
use wf_market::types::{Order, OrderList, Properties as WfmProperties};

use crate::cache::types::ItemPriceInfo;

const MODEL_VERSION: &str = "ml_bandit_v3";
const STORE_VERSION: u32 = 3;
const OUTCOME_HALF_LIFE_DAYS: f64 = 30.0;
const CONFIDENCE_PRIOR_UNITS: f64 = 14.0;
const EXPIRED_ORDER_GRACE_HOURS: f64 = 2.0;
const MAX_DECISIONS: usize = 50_000;

const DECISION_STORE_FILE: &str = "algoframe_learning_decisions.json";
const INSPECTOR_FILE: &str = "algoframe_learning_inspector.json";

const MAX_PREDICTION_BIAS_WEIGHT: f64 = 0.55;
const MAX_REALIZED_PROFIT_WEIGHT: f64 = 0.30;
const MAX_PERSONAL_PRICE_WEIGHT: f64 = 0.10;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MarketRegime {
    Normal,
    Moving,
    Shock,
}

impl MarketRegime {
    fn confidence_multiplier(self) -> f64 {
        match self {
            Self::Normal => 1.0,
            Self::Moving => 0.55,
            Self::Shock => 0.25,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Moving => "moving",
            Self::Shock => "shock",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BuyAction {
    Match,
    PlusOne,
    PlusTwo,
}

impl BuyAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::PlusOne => "plus_1",
            Self::PlusTwo => "plus_2",
        }
    }

    fn offset(self) -> i64 {
        match self {
            Self::Match => 0,
            Self::PlusOne => 1,
            Self::PlusTwo => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SellAction {
    Hold,
    MatchFloor,
    UndercutOne,
    UndercutTwo,
}

impl SellAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::MatchFloor => "match_floor",
            Self::UndercutOne => "undercut_1",
            Self::UndercutTwo => "undercut_2",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStatus {
    Open,
    Filled,
    Completed,
    Expired,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct DecisionFeatures {
    pub liquidity: f64,
    pub spread_percent: f64,
    pub volatility: f64,
    pub inventory_pressure: f64,
    pub week_shift: f64,
    pub learning_confidence: f64,
    pub market_depth: i64,
    pub current_price: i64,
    pub expected_sell: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub id: String,
    pub model_version: String,
    pub side: TradeSide,
    pub wfm_id: String,
    pub item_key: String,
    pub context_key: String,
    pub action: String,

    pub price: i64,
    pub quantity: i64,
    pub capital: f64,

    pub predicted_profit: f64,
    pub predicted_fill_hours: f64,
    pub predicted_sell_hours: f64,
    pub predicted_reward: f64,

    pub features: DecisionFeatures,

    pub status: DecisionStatus,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub filled_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,

    pub actual_profit: Option<f64>,
    pub actual_fill_hours: Option<f64>,
    pub actual_cycle_hours: Option<f64>,
    pub actual_reward: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DecisionStore {
    version: u32,
    decisions: Vec<DecisionRecord>,
}

impl Default for DecisionStore {
    fn default() -> Self {
        Self {
            version: STORE_VERSION,
            decisions: vec![],
        }
    }
}

impl DecisionStore {
    fn path() -> PathBuf {
        PathBuf::from(utils::get_base_path()).join(DECISION_STORE_FILE)
    }

    fn inspector_path() -> PathBuf {
        PathBuf::from(utils::get_base_path()).join(INSPECTOR_FILE)
    }

    fn load() -> Self {
        let path = Self::path();
        if !path.exists() {
            return Self::default();
        }

        match fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Self>(&text).ok())
        {
            Some(mut store) => {
                if store.version != STORE_VERSION {
                    store.version = STORE_VERSION;
                }
                store
            }
            None => Self::default(),
        }
    }

    fn save(&mut self) -> Result<(), Error> {
        if self.decisions.len() > MAX_DECISIONS {
            self.decisions.sort_by_key(|decision| decision.created_at);
            let drain = self.decisions.len() - MAX_DECISIONS;
            self.decisions.drain(0..drain);
        }

        let path = Self::path();
        let temp = path.with_extension("json.tmp");

        let content = serde_json::to_string_pretty(self).map_err(|error| {
            Error::new(
                "AlgoFrame:Learning:SerializeStore",
                error.to_string(),
                get_location!(),
            )
        })?;

        fs::write(&temp, content).map_err(|error| {
            Error::new(
                "AlgoFrame:Learning:WriteStore",
                error.to_string(),
                get_location!(),
            )
        })?;

        if path.exists() {
            let _ = fs::remove_file(&path);
        }

        fs::rename(&temp, &path).map_err(|error| {
            Error::new(
                "AlgoFrame:Learning:CommitStore",
                error.to_string(),
                get_location!(),
            )
        })?;

        Ok(())
    }

    fn find_mut(&mut self, id: &str) -> Option<&mut DecisionRecord> {
        self.decisions.iter_mut().find(|decision| decision.id == id)
    }

    fn find(&self, id: &str) -> Option<&DecisionRecord> {
        self.decisions.iter().find(|decision| decision.id == id)
    }
}

#[derive(Clone, Copy, Debug)]
struct WeightedSample {
    value: f64,
    weight: f64,
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
struct LearningAccumulator {
    purchase_units: i64,
    sold_units: i64,
    matched_units: i64,
    open_units: i64,
    trade_count: usize,

    profit: Vec<WeightedSample>,
    roi: Vec<WeightedSample>,
    hold_hours: Vec<WeightedSample>,
    sale_price: Vec<WeightedSample>,
    buy_fill_hours: Vec<WeightedSample>,
    sell_fill_hours: Vec<WeightedSample>,
    open_age_hours: Vec<WeightedSample>,
    prediction_error: Vec<WeightedSample>,
    prediction_abs_error: Vec<WeightedSample>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LearnedProfile {
    pub purchase_units: i64,
    pub sold_units: i64,
    pub matched_units: i64,
    pub open_units: i64,
    pub trade_count: usize,

    pub avg_profit: f64,
    pub avg_roi: f64,
    pub avg_hold_hours: f64,
    pub avg_sale_price: f64,
    pub avg_buy_fill_hours: f64,
    pub avg_sell_fill_hours: f64,
    pub avg_open_age_hours: f64,

    pub prediction_bias: f64,
    pub prediction_mae: f64,
    pub prediction_accuracy: f64,

    pub sell_through: f64,
    pub profit_stability: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug, Default)]
struct LearnedBook {
    item_profiles: HashMap<String, LearnedProfile>,
    tag_profiles: HashMap<String, LearnedProfile>,
    global: LearnedProfile,
    transaction_count: usize,
}

#[derive(Clone, Debug, Default)]
struct ContextBanditStats {
    trials: f64,
    successes: f64,
    failures: f64,
    reward_samples: Vec<WeightedSample>,
    exposure_hours: f64,
}

impl ContextBanditStats {
    fn reward_mean(&self) -> f64 {
        robust_weighted_mean(&self.reward_samples, 0.0)
    }

    fn reward_std(&self) -> f64 {
        let mean = self.reward_mean();
        let total = total_weight(&self.reward_samples);
        if total <= 0.0 {
            return 0.25;
        }

        let variance = self
            .reward_samples
            .iter()
            .filter(|sample| sample.weight > 0.0)
            .map(|sample| (sample.value - mean).powi(2) * sample.weight)
            .sum::<f64>()
            / total;

        variance.sqrt().max(0.05)
    }

    fn hazard_per_hour(&self) -> f64 {
        if self.exposure_hours <= 0.0 {
            0.0
        } else {
            (self.successes / self.exposure_hours).max(0.0)
        }
    }

    fn expected_fill_hours(&self, fallback: f64) -> f64 {
        let hazard = self.hazard_per_hour();
        if hazard > 0.0 {
            (1.0 / hazard).clamp(0.05, 720.0)
        } else {
            fallback
        }
    }
}

#[derive(Clone, Debug)]
pub struct BuyPlan {
    pub eligible: bool,
    pub action: BuyAction,
    pub context_key: String,

    pub recommended_buy_price: i64,
    pub safe_max_buy_price: i64,
    pub expected_sell_price: f64,

    pub expected_profit: f64,
    pub predicted_fill_hours: f64,
    pub predicted_sell_hours: f64,
    pub predicted_reward: f64,

    pub regime: MarketRegime,
    pub learning_confidence: f64,
    pub prediction_accuracy: f64,
    pub inventory_pressure: f64,
    pub allocation_value: f64,
    pub score: f64,

    pub sell_outliers_removed: usize,
    pub buy_outliers_removed: usize,
}

impl Default for BuyPlan {
    fn default() -> Self {
        Self {
            eligible: false,
            action: BuyAction::Match,
            context_key: String::new(),
            recommended_buy_price: 0,
            safe_max_buy_price: 0,
            expected_sell_price: 0.0,
            expected_profit: 0.0,
            predicted_fill_hours: 0.0,
            predicted_sell_hours: 0.0,
            predicted_reward: 0.0,
            regime: MarketRegime::Normal,
            learning_confidence: 0.0,
            prediction_accuracy: 0.5,
            inventory_pressure: 0.0,
            allocation_value: 0.0,
            score: 0.0,
            sell_outliers_removed: 0,
            buy_outliers_removed: 0,
        }
    }
}

impl BuyPlan {
    pub fn apply_properties(&self, properties: &mut WfmProperties) {
        ensure_order_started_at(properties);

        properties.set_property_value("algoframe_model", MODEL_VERSION);
        properties.set_property_value("algoframe_context_key", &self.context_key);
        properties.set_property_value("algoframe_action", self.action.as_str());
        properties.set_property_value("algoframe_regime", self.regime.as_str());
        properties.set_property_value("algoframe_expected_sell", self.expected_sell_price);
        properties.set_property_value("algoframe_expected_profit", self.expected_profit);
        properties.set_property_value("algoframe_predicted_fill_hours", self.predicted_fill_hours);
        properties.set_property_value("algoframe_predicted_sell_hours", self.predicted_sell_hours);
        properties.set_property_value("algoframe_predicted_reward", self.predicted_reward);
        properties.set_property_value(
            "algoframe_learning_confidence",
            self.learning_confidence,
        );
        properties.set_property_value(
            "algoframe_prediction_accuracy",
            self.prediction_accuracy,
        );
        properties.set_property_value(
            "algoframe_inventory_pressure",
            self.inventory_pressure,
        );
        properties.set_property_value("algoframe_safe_max_buy", self.safe_max_buy_price);
        properties.set_property_value("algoframe_score", self.score);
        properties.set_property_value("allocation_value", self.allocation_value);
        properties.set_property_value(
            "algoframe_sell_outliers_removed",
            self.sell_outliers_removed as i64,
        );
        properties.set_property_value(
            "algoframe_buy_outliers_removed",
            self.buy_outliers_removed as i64,
        );
    }
}

#[derive(Clone, Debug)]
pub struct SellPlan {
    pub eligible: bool,
    pub action: SellAction,
    pub context_key: String,

    pub recommended_sell_price: i64,
    pub robust_market_floor: i64,
    pub expected_profit: f64,

    pub predicted_fill_hours: f64,
    pub predicted_reward: f64,

    pub regime: MarketRegime,
    pub learning_confidence: f64,
    pub prediction_accuracy: f64,
    pub inventory_pressure: f64,
    pub order_age_hours: f64,

    pub outliers_removed: usize,
}

impl Default for SellPlan {
    fn default() -> Self {
        Self {
            eligible: false,
            action: SellAction::MatchFloor,
            context_key: String::new(),
            recommended_sell_price: 0,
            robust_market_floor: 0,
            expected_profit: 0.0,
            predicted_fill_hours: 0.0,
            predicted_reward: 0.0,
            regime: MarketRegime::Normal,
            learning_confidence: 0.0,
            prediction_accuracy: 0.5,
            inventory_pressure: 0.0,
            order_age_hours: 0.0,
            outliers_removed: 0,
        }
    }
}

impl SellPlan {
    pub fn apply_properties(&self, properties: &mut WfmProperties) {
        ensure_order_started_at(properties);

        properties.set_property_value("algoframe_model", MODEL_VERSION);
        properties.set_property_value("algoframe_context_key", &self.context_key);
        properties.set_property_value("algoframe_action", self.action.as_str());
        properties.set_property_value("algoframe_regime", self.regime.as_str());
        properties.set_property_value(
            "algoframe_recommended_sell",
            self.recommended_sell_price,
        );
        properties.set_property_value(
            "algoframe_robust_sell_floor",
            self.robust_market_floor,
        );
        properties.set_property_value("algoframe_expected_profit", self.expected_profit);
        properties.set_property_value("algoframe_predicted_fill_hours", self.predicted_fill_hours);
        properties.set_property_value("algoframe_predicted_reward", self.predicted_reward);
        properties.set_property_value(
            "algoframe_learning_confidence",
            self.learning_confidence,
        );
        properties.set_property_value(
            "algoframe_prediction_accuracy",
            self.prediction_accuracy,
        );
        properties.set_property_value(
            "algoframe_inventory_pressure",
            self.inventory_pressure,
        );
        properties.set_property_value("algoframe_order_age_hours", self.order_age_hours);
        properties.set_property_value(
            "algoframe_sell_outliers_removed",
            self.outliers_removed as i64,
        );
    }
}

#[derive(Clone, Debug)]
pub struct LearningEngine {
    learned: LearnedBook,
    store: DecisionStore,
    bandit: HashMap<(TradeSide, String, String), ContextBanditStats>,
    dirty: bool,
}

impl LearningEngine {
    pub async fn load(
        db: &DatabaseConnection,
        my_orders: &OrderList<Order>,
    ) -> Result<Self, Error> {
        let transactions =
            TransactionQuery::get_all(db, TransactionPaginationQueryDto::new(1, -1)).await?;

        let mut transactions = transactions.results;
        transactions.sort_by_key(|transaction| transaction.created_at);

        let mut store = DecisionStore::load();

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

        reconcile_decision_outcomes(
            &mut store,
            &transactions,
            &active_decision_ids,
        );

        let learned = LearnedBook::from_transactions(&transactions);
        let bandit = build_bandit(&store);

        Ok(Self {
            learned,
            store,
            bandit,
            dirty: true,
        })
    }

    pub fn transaction_count(&self) -> usize {
        self.learned.transaction_count
    }

    pub fn decision_count(&self) -> usize {
        self.store.decisions.len()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan_buy(
        &self,
        wfm_id: &str,
        sub_type: &Option<SubType>,
        tags: &[String],
        price: &ItemPriceInfo,
        buy_prices: &[i64],
        sell_prices: &[i64],
        minimum_profit: i64,
        minimum_margin_percent: i64,
        quantity: i64,
    ) -> BuyPlan {
        let context = self.learned.contextual_profile(wfm_id, sub_type, tags);

        let historical = historical_center(price);
        let Some(sell_level) = robust_floor(sell_prices, historical) else {
            return BuyPlan::default();
        };
        let Some(buy_level) = robust_ceiling(buy_prices) else {
            return BuyPlan::default();
        };

        let live_mid =
            (sell_level.price as f64 + buy_level.price as f64) / 2.0;
        let regime = detect_regime(live_mid, historical, price.week_price_shift);
        let effective_confidence =
            context.profile.confidence * regime.confidence_multiplier();

        let live_sell =
            sell_level.price as f64 * 0.88 + sell_level.cluster_median as f64 * 0.12;

        let mut expected_sell = if historical > 0.0 {
            live_sell * 0.90 + historical * 0.10
        } else {
            live_sell
        };

        if context.profile.avg_sale_price > 0.0
            && effective_confidence > 0.0
        {
            let learned_price = context.profile.avg_sale_price.clamp(
                sell_level.price as f64 * 0.75,
                sell_level.cluster_median as f64 * 1.20,
            );
            let weight =
                MAX_PERSONAL_PRICE_WEIGHT * effective_confidence;

            expected_sell =
                expected_sell * (1.0 - weight) + learned_price * weight;
        }

        expected_sell = expected_sell.clamp(
            sell_level.price as f64 * 0.90,
            sell_level.cluster_median.max(sell_level.price) as f64 * 1.05,
        );

        let minimum_profit = normalize_disabled(minimum_profit);
        let safe_by_profit =
            (expected_sell.floor() as i64 - minimum_profit).max(0);

        let safe_by_margin = if minimum_margin_percent <= -1 {
            safe_by_profit
        } else {
            let margin =
                minimum_margin_percent.max(0) as f64 / 100.0;
            (expected_sell / (1.0 + margin)).floor() as i64
        };

        let safe_max_buy_price = safe_by_profit.min(safe_by_margin);
        if safe_max_buy_price < 1 || buy_level.price > safe_max_buy_price {
            return BuyPlan {
                expected_sell_price: expected_sell,
                safe_max_buy_price,
                regime,
                learning_confidence: effective_confidence,
                prediction_accuracy: context.profile.prediction_accuracy,
                sell_outliers_removed: sell_level.removed,
                buy_outliers_removed: buy_level.removed,
                ..Default::default()
            };
        }

        let liquidity = liquidity_score(price.volume);
        let volatility = price_volatility(price);
        let pressure = inventory_pressure(&context.profile);
        let spread_percent = if expected_sell > 0.0 {
            ((expected_sell - buy_level.price as f64) / expected_sell)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        let context_key = make_context_key(
            TradeSide::Buy,
            tags,
            liquidity,
            spread_percent,
            volatility,
            pressure,
            regime,
        );

        let fallback_sell_hours = if context.profile.avg_hold_hours > 0.0 {
            context.profile.avg_hold_hours
        } else if context.profile.avg_sell_fill_hours > 0.0 {
            context.profile.avg_sell_fill_hours
        } else {
            (24.0 / (0.20 + 2.80 * liquidity)).clamp(0.5, 120.0)
        };

        let actions = [
            BuyAction::Match,
            BuyAction::PlusOne,
            BuyAction::PlusTwo,
        ];

        let mut candidates = Vec::new();

        for action in actions {
            let candidate_price = buy_level.price + action.offset();
            if candidate_price <= 0 || candidate_price > safe_max_buy_price {
                continue;
            }

            let raw_market_profit =
                expected_sell - candidate_price as f64;

            let prediction_weight =
                MAX_PREDICTION_BIAS_WEIGHT
                    * effective_confidence
                    * context.profile.prediction_accuracy;

            let realized_weight =
                MAX_REALIZED_PROFIT_WEIGHT * effective_confidence;

            let mut expected_profit =
                raw_market_profit
                    + context.profile.prediction_bias * prediction_weight;

            if context.profile.avg_profit > 0.0 {
                expected_profit = expected_profit * (1.0 - realized_weight)
                    + context.profile.avg_profit * realized_weight;
            }

            if raw_market_profit > 0.0 {
                expected_profit = expected_profit.clamp(
                    raw_market_profit * 0.50,
                    raw_market_profit * 1.45 + 2.0,
                );
            }

            let roi = expected_profit / candidate_price.max(1) as f64;
            let required_roi = if minimum_margin_percent <= -1 {
                0.0
            } else {
                minimum_margin_percent.max(0) as f64 / 100.0
            };

            if expected_profit < minimum_profit as f64 || roi < required_roi {
                continue;
            }

            let stats = self.bandit_stats(
                TradeSide::Buy,
                &context_key,
                action.as_str(),
            );

            let market_fill_fallback = match action {
                BuyAction::Match => 8.0 / (0.25 + liquidity),
                BuyAction::PlusOne => 5.0 / (0.25 + liquidity),
                BuyAction::PlusTwo => 3.5 / (0.25 + liquidity),
            }
            .clamp(0.15, 96.0);

            let predicted_fill_hours =
                stats.expected_fill_hours(market_fill_fallback);

            let predicted_cycle_hours =
                (predicted_fill_hours + fallback_sell_hours).max(0.25);

            let capital = candidate_price.max(1) as f64
                * quantity.max(1) as f64;

            let predicted_reward = normalized_reward(
                expected_profit * quantity.max(1) as f64,
                capital,
                predicted_cycle_hours,
            );

            let posterior_utility = sample_bandit_utility(
                stats,
                action_prior_success(action),
                predicted_reward,
            );

            let inventory_multiplier =
                (1.0 - pressure * 0.70).clamp(0.20, 1.0);
            let stability =
                (1.0 / (1.0 + volatility)).clamp(0.15, 1.0);
            let accuracy =
                (0.50 + 0.50 * context.profile.prediction_accuracy)
                    .clamp(0.50, 1.0);

            let allocation_value = expected_profit.max(0.0)
                * (24.0 / predicted_cycle_hours)
                * quantity.max(1) as f64
                * inventory_multiplier
                * stability
                * accuracy;

            candidates.push((
                posterior_utility,
                action,
                candidate_price,
                expected_profit,
                predicted_fill_hours,
                predicted_reward,
                allocation_value,
                roi,
            ));
        }

        let Some(best) = candidates.into_iter().max_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
        }) else {
            return BuyPlan {
                expected_sell_price: expected_sell,
                safe_max_buy_price,
                context_key,
                regime,
                learning_confidence: effective_confidence,
                prediction_accuracy: context.profile.prediction_accuracy,
                inventory_pressure: pressure,
                sell_outliers_removed: sell_level.removed,
                buy_outliers_removed: buy_level.removed,
                ..Default::default()
            };
        };

        let (_, action, price_out, expected_profit, predicted_fill_hours, predicted_reward, allocation_value, roi) = best;

        let eligible = pressure < 0.95 && predicted_reward > 0.0;
        let score = if price_out > 0 {
            allocation_value
                / (price_out as f64 * quantity.max(1) as f64)
                * 100.0
        } else {
            0.0
        };

        BuyPlan {
            eligible,
            action,
            context_key,
            recommended_buy_price: price_out,
            safe_max_buy_price,
            expected_sell_price: expected_sell,
            expected_profit,
            predicted_fill_hours,
            predicted_sell_hours: fallback_sell_hours,
            predicted_reward,
            regime,
            learning_confidence: effective_confidence,
            prediction_accuracy: context.profile.prediction_accuracy,
            inventory_pressure: pressure,
            allocation_value,
            score: score * (1.0 + roi.clamp(0.0, 1.5) * 0.10),
            sell_outliers_removed: sell_level.removed,
            buy_outliers_removed: buy_level.removed,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan_sell(
        &self,
        wfm_id: &str,
        sub_type: &Option<SubType>,
        tags: &[String],
        price: &ItemPriceInfo,
        sell_prices: &[i64],
        bought_price: i64,
        minimum_profit: i64,
        current_order_price: i64,
        current_order_age_hours: f64,
        quantity: i64,
    ) -> SellPlan {
        let context = self.learned.contextual_profile(wfm_id, sub_type, tags);
        let historical = historical_center(price);

        let Some(floor) = robust_floor(sell_prices, historical) else {
            return SellPlan::default();
        };

        let regime = detect_regime(
            floor.cluster_median as f64,
            historical,
            price.week_price_shift,
        );

        let effective_confidence =
            context.profile.confidence * regime.confidence_multiplier();

        let minimum_profit = normalize_disabled(minimum_profit);
        let minimum_allowed =
            bought_price.saturating_add(minimum_profit).max(1);

        let liquidity = liquidity_score(price.volume);
        let volatility = price_volatility(price);
        let pressure = inventory_pressure(&context.profile);

        let spread_percent = if floor.cluster_median > 0 {
            ((floor.cluster_median - floor.price) as f64
                / floor.cluster_median as f64)
                .abs()
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        let context_key = make_context_key(
            TradeSide::Sell,
            tags,
            liquidity,
            spread_percent,
            volatility,
            pressure,
            regime,
        );

        let mut candidates: Vec<(f64, SellAction, i64, f64, f64, f64)> =
            Vec::new();

        let action_prices = [
            (SellAction::Hold, current_order_price),
            (SellAction::MatchFloor, floor.price),
            (SellAction::UndercutOne, floor.price.saturating_sub(1)),
            (SellAction::UndercutTwo, floor.price.saturating_sub(2)),
        ];

        let mut seen_prices = HashSet::new();

        for (action, raw_price) in action_prices {
            if raw_price <= 0 {
                continue;
            }

            let candidate_price = raw_price.max(minimum_allowed);
            if !seen_prices.insert(candidate_price) {
                continue;
            }

            let expected_profit =
                (candidate_price - bought_price).max(0) as f64;

            if expected_profit < minimum_profit as f64 {
                continue;
            }

            let stats = self.bandit_stats(
                TradeSide::Sell,
                &context_key,
                action.as_str(),
            );

            let market_fallback = match action {
                SellAction::Hold => {
                    if context.profile.avg_sell_fill_hours > 0.0 {
                        context.profile.avg_sell_fill_hours
                    } else {
                        18.0 / (0.25 + liquidity)
                    }
                }
                SellAction::MatchFloor => 10.0 / (0.25 + liquidity),
                SellAction::UndercutOne => 7.0 / (0.25 + liquidity),
                SellAction::UndercutTwo => 5.0 / (0.25 + liquidity),
            }
            .clamp(0.10, 120.0);

            let predicted_fill_hours =
                stats.expected_fill_hours(market_fallback);

            let capital = bought_price.max(1) as f64
                * quantity.max(1) as f64;

            let predicted_reward = normalized_reward(
                expected_profit * quantity.max(1) as f64,
                capital,
                predicted_fill_hours.max(0.10),
            );

            let mut posterior_utility = sample_bandit_utility(
                stats,
                sell_action_prior_success(action),
                predicted_reward,
            );

            // If an order is already materially stale relative to the action's
            // expected fill time, reward more aggressive actions.
            let stale_ratio =
                current_order_age_hours / predicted_fill_hours.max(0.25);

            posterior_utility *= match action {
                SellAction::Hold if stale_ratio >= 1.5 => 0.70,
                SellAction::UndercutOne if stale_ratio >= 1.5 => 1.10,
                SellAction::UndercutTwo if stale_ratio >= 2.5 => 1.20,
                _ => 1.0,
            };

            posterior_utility *= (1.0 + pressure * match action {
                SellAction::Hold => -0.30,
                SellAction::MatchFloor => 0.00,
                SellAction::UndercutOne => 0.12,
                SellAction::UndercutTwo => 0.20,
            })
            .max(0.50);

            candidates.push((
                posterior_utility,
                action,
                candidate_price,
                expected_profit,
                predicted_fill_hours,
                predicted_reward,
            ));
        }

        let Some(best) = candidates.into_iter().max_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
        }) else {
            return SellPlan {
                robust_market_floor: floor.price,
                context_key,
                regime,
                learning_confidence: effective_confidence,
                prediction_accuracy: context.profile.prediction_accuracy,
                inventory_pressure: pressure,
                order_age_hours: current_order_age_hours,
                outliers_removed: floor.removed,
                ..Default::default()
            };
        };

        let (_, action, recommended_sell_price, expected_profit, predicted_fill_hours, predicted_reward) = best;

        SellPlan {
            eligible: true,
            action,
            context_key,
            recommended_sell_price,
            robust_market_floor: floor.price,
            expected_profit,
            predicted_fill_hours,
            predicted_reward,
            regime,
            learning_confidence: effective_confidence,
            prediction_accuracy: context.profile.prediction_accuracy,
            inventory_pressure: pressure,
            order_age_hours: current_order_age_hours,
            outliers_removed: floor.removed,
        }
    }

    pub fn attach_buy_decision(
        &mut self,
        properties: &mut WfmProperties,
        wfm_id: &str,
        sub_type: &Option<SubType>,
        quantity: i64,
        plan: &BuyPlan,
    ) {
        if !plan.eligible {
            return;
        }

        plan.apply_properties(properties);

        let features = DecisionFeatures {
            liquidity: properties.get_property_value("velocity", 0.0_f64),
            spread_percent: properties.get_property_value("spread_percent", 0.0_f64),
            volatility: 0.0,
            inventory_pressure: plan.inventory_pressure,
            week_shift: 0.0,
            learning_confidence: plan.learning_confidence,
            market_depth: 0,
            current_price: plan.recommended_buy_price,
            expected_sell: plan.expected_sell_price,
        };

        self.attach_decision(
            properties,
            TradeSide::Buy,
            wfm_id,
            sub_type,
            &plan.context_key,
            plan.action.as_str(),
            plan.recommended_buy_price,
            quantity,
            plan.expected_profit,
            plan.predicted_fill_hours,
            plan.predicted_sell_hours,
            plan.predicted_reward,
            features,
        );
    }

    pub fn attach_sell_decision(
        &mut self,
        properties: &mut WfmProperties,
        wfm_id: &str,
        sub_type: &Option<SubType>,
        quantity: i64,
        bought_price: i64,
        plan: &SellPlan,
    ) {
        if !plan.eligible {
            return;
        }

        properties.set_property_value("algoframe_bought_price", bought_price);
        plan.apply_properties(properties);

        let features = DecisionFeatures {
            liquidity: properties.get_property_value("velocity", 0.0_f64),
            spread_percent: properties.get_property_value("spread_percent", 0.0_f64),
            volatility: 0.0,
            inventory_pressure: plan.inventory_pressure,
            week_shift: 0.0,
            learning_confidence: plan.learning_confidence,
            market_depth: 0,
            current_price: plan.recommended_sell_price,
            expected_sell: plan.recommended_sell_price as f64,
        };

        self.attach_decision(
            properties,
            TradeSide::Sell,
            wfm_id,
            sub_type,
            &plan.context_key,
            plan.action.as_str(),
            plan.recommended_sell_price,
            quantity,
            plan.expected_profit,
            plan.predicted_fill_hours,
            0.0,
            plan.predicted_reward,
            features,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn attach_decision(
        &mut self,
        properties: &mut WfmProperties,
        side: TradeSide,
        wfm_id: &str,
        sub_type: &Option<SubType>,
        context_key: &str,
        action: &str,
        price: i64,
        quantity: i64,
        predicted_profit: f64,
        predicted_fill_hours: f64,
        predicted_sell_hours: f64,
        predicted_reward: f64,
        features: DecisionFeatures,
    ) {
        let existing_id: String = properties.get_property_value(
            "algoframe_decision_id",
            String::new(),
        );

        if !existing_id.is_empty() {
            if let Some(existing) = self.store.find(&existing_id) {
                if existing.status == DecisionStatus::Open
                    && existing.side == side
                    && existing.context_key == context_key
                    && existing.action == action
                    && existing.price == price
                {
                    return;
                }
            }
        }

        let id = Uuid::new_v4().to_string();
        let now = Utc::now();

        self.store.decisions.push(DecisionRecord {
            id: id.clone(),
            model_version: MODEL_VERSION.to_string(),
            side,
            wfm_id: wfm_id.to_string(),
            item_key: item_key(wfm_id, sub_type),
            context_key: context_key.to_string(),
            action: action.to_string(),
            price,
            quantity: quantity.max(1),
            capital: match side {
                TradeSide::Buy => price.max(1) as f64 * quantity.max(1) as f64,
                TradeSide::Sell => properties
                    .get_property_value::<i64>("algoframe_bought_price", price)
                    .max(1) as f64
                    * quantity.max(1) as f64,
            },
            predicted_profit,
            predicted_fill_hours,
            predicted_sell_hours,
            predicted_reward,
            features,
            status: DecisionStatus::Open,
            created_at: now,
            updated_at: now,
            filled_at: None,
            completed_at: None,
            actual_profit: None,
            actual_fill_hours: None,
            actual_cycle_hours: None,
            actual_reward: None,
        });

        properties.set_property_value("algoframe_decision_id", id);
        properties.set_property_value(
            "algoframe_order_started_at",
            now.to_rfc3339(),
        );

        self.dirty = true;
    }

    pub fn flush(&mut self) -> Result<(), Error> {
        if self.dirty {
            self.store.save()?;
            self.dirty = false;
        }

        self.write_inspector()
    }

    fn write_inspector(&self) -> Result<(), Error> {
        let completed: Vec<&DecisionRecord> = self
            .store
            .decisions
            .iter()
            .filter(|decision| decision.status == DecisionStatus::Completed)
            .collect();

        let expired = self
            .store
            .decisions
            .iter()
            .filter(|decision| decision.status == DecisionStatus::Expired)
            .count();

        let open = self
            .store
            .decisions
            .iter()
            .filter(|decision| decision.status == DecisionStatus::Open)
            .count();

        let avg_reward = if completed.is_empty() {
            0.0
        } else {
            completed
                .iter()
                .filter_map(|decision| decision.actual_reward)
                .sum::<f64>()
                / completed.len() as f64
        };

        let avg_prediction_error = {
            let errors: Vec<f64> = completed
                .iter()
                .filter_map(|decision| {
                    decision.actual_profit.map(|actual| {
                        (actual - decision.predicted_profit).abs()
                    })
                })
                .collect();

            if errors.is_empty() {
                0.0
            } else {
                errors.iter().sum::<f64>() / errors.len() as f64
            }
        };

        let mut item_profiles: Vec<serde_json::Value> = self
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

        item_profiles.sort_by(|a, b| {
            let av = a
                .get("confidence")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0);
            let bv = b
                .get("confidence")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0);

            bv.partial_cmp(&av).unwrap_or(Ordering::Equal)
        });
        item_profiles.truncate(100);

        let inspector = serde_json::json!({
            "model": MODEL_VERSION,
            "updated_at": Utc::now().to_rfc3339(),
            "transactions_learned": self.learned.transaction_count,
            "decisions_total": self.store.decisions.len(),
            "decisions_completed": completed.len(),
            "decisions_expired": expired,
            "decisions_open": open,
            "average_reward": avg_reward,
            "average_absolute_prediction_error": avg_prediction_error,
            "decision_store_path": DecisionStore::path(),
            "top_learned_items": item_profiles,
        });

        let path = DecisionStore::inspector_path();
        let temp = path.with_extension("json.tmp");

        fs::write(
            &temp,
            serde_json::to_string_pretty(&inspector).unwrap_or_default(),
        )
        .map_err(|error| {
            Error::new(
                "AlgoFrame:Learning:WriteInspector",
                error.to_string(),
                get_location!(),
            )
        })?;

        if path.exists() {
            let _ = fs::remove_file(&path);
        }

        fs::rename(&temp, &path).map_err(|error| {
            Error::new(
                "AlgoFrame:Learning:CommitInspector",
                error.to_string(),
                get_location!(),
            )
        })?;

        Ok(())
    }

    fn bandit_stats(
        &self,
        side: TradeSide,
        context_key: &str,
        action: &str,
    ) -> &ContextBanditStats {
        static EMPTY: std::sync::OnceLock<ContextBanditStats> =
            std::sync::OnceLock::new();

        self.bandit
            .get(&(side, context_key.to_string(), action.to_string()))
            .unwrap_or_else(|| EMPTY.get_or_init(ContextBanditStats::default))
    }
}

#[derive(Clone, Debug)]
struct ContextualProfile {
    profile: LearnedProfile,
}

impl LearnedBook {
    fn from_transactions(transactions: &[TransactionModel]) -> Self {
        let mut grouped: HashMap<String, Vec<TransactionModel>> = HashMap::new();

        for transaction in transactions {
            grouped
                .entry(item_key(&transaction.wfm_id, &transaction.sub_type))
                .or_default()
                .push(transaction.clone());
        }

        let mut item_profiles = HashMap::new();
        let mut tag_accumulators: HashMap<String, LearningAccumulator> =
            HashMap::new();
        let mut global_accumulator = LearningAccumulator::default();

        for (key, mut item_transactions) in grouped {
            item_transactions.sort_by_key(|transaction| transaction.created_at);
            let tags = collect_tags(&item_transactions);

            let accumulator =
                LearningAccumulator::from_transactions(&item_transactions);

            global_accumulator.merge(&accumulator);

            for tag in tags {
                tag_accumulators
                    .entry(tag)
                    .or_default()
                    .merge(&accumulator);
            }

            item_profiles.insert(key, accumulator.profile());
        }

        let tag_profiles = tag_accumulators
            .into_iter()
            .map(|(tag, accumulator)| (tag, accumulator.profile()))
            .collect();

        Self {
            item_profiles,
            tag_profiles,
            global: global_accumulator.profile(),
            transaction_count: transactions.len(),
        }
    }

    fn contextual_profile(
        &self,
        wfm_id: &str,
        sub_type: &Option<SubType>,
        tags: &[String],
    ) -> ContextualProfile {
        let item = self
            .item_profiles
            .get(&item_key(wfm_id, sub_type))
            .cloned()
            .unwrap_or_default();

        let tag_profiles: Vec<LearnedProfile> = tags
            .iter()
            .filter_map(|tag| {
                self.tag_profiles
                    .get(&normalize_tag(tag))
                    .cloned()
            })
            .collect();

        let tag = average_profiles(&tag_profiles);
        let mut prior = self.global.clone();

        if let Some(tag_profile) = tag {
            let tag_weight = tag_profile.confidence.clamp(0.10, 0.60);
            prior = blend_profiles(&prior, &tag_profile, tag_weight);
        }

        let mut profile = if item.confidence > 0.0 {
            blend_profiles(&prior, &item, item.confidence)
        } else {
            prior
        };

        profile.purchase_units = item.purchase_units;
        profile.sold_units = item.sold_units;
        profile.matched_units = item.matched_units;
        profile.open_units = item.open_units;
        profile.trade_count = item.trade_count;
        profile.confidence = item.confidence;

        ContextualProfile { profile }
    }
}

impl LearningAccumulator {
    fn from_transactions(transactions: &[TransactionModel]) -> Self {
        let mut result = Self::default();
        let mut lots: VecDeque<PurchaseLot> = VecDeque::new();

        for transaction in transactions {
            if transaction.quantity <= 0 || transaction.price <= 0 {
                continue;
            }

            result.trade_count += 1;

            let quantity = transaction.quantity;
            let unit_price = transaction.price as f64 / quantity as f64;
            let weight =
                recency_weight(transaction.created_at) * quantity as f64;

            if transaction.transaction_type == TransactionType::Purchase {
                result.purchase_units += quantity;

                if let Some(hours) =
                    transaction_order_age_hours(transaction)
                {
                    result.buy_fill_hours.push(WeightedSample {
                        value: hours,
                        weight,
                    });
                }

                lots.push_back(PurchaseLot {
                    remaining: quantity,
                    unit_price,
                    at: transaction.created_at,
                    decision_id: transaction_property_string(
                        transaction,
                        "algoframe_decision_id",
                    ),
                    predicted_profit: transaction_property_f64(
                        transaction,
                        "algoframe_expected_profit",
                    ),
                });

                continue;
            }

            result.sold_units += quantity;

            result.sale_price.push(WeightedSample {
                value: unit_price,
                weight,
            });

            if let Some(hours) =
                transaction_order_age_hours(transaction)
            {
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
                    (transaction.created_at - lot.at).num_seconds().max(0)
                        as f64
                        / 3600.0;

                let matched_weight =
                    recency_weight(transaction.created_at) * matched as f64;

                result.matched_units += matched;

                result.profit.push(WeightedSample {
                    value: profit,
                    weight: matched_weight,
                });
                result.roi.push(WeightedSample {
                    value: roi,
                    weight: matched_weight,
                });
                result.hold_hours.push(WeightedSample {
                    value: hold_hours,
                    weight: matched_weight,
                });

                if let Some(predicted) = lot.predicted_profit {
                    let error = profit - predicted;
                    result.prediction_error.push(WeightedSample {
                        value: error,
                        weight: matched_weight,
                    });
                    result.prediction_abs_error.push(WeightedSample {
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

            let age_hours =
                (Utc::now() - lot.at).num_seconds().max(0) as f64
                    / 3600.0;

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

        self.profit.extend_from_slice(&other.profit);
        self.roi.extend_from_slice(&other.roi);
        self.hold_hours.extend_from_slice(&other.hold_hours);
        self.sale_price.extend_from_slice(&other.sale_price);
        self.buy_fill_hours
            .extend_from_slice(&other.buy_fill_hours);
        self.sell_fill_hours
            .extend_from_slice(&other.sell_fill_hours);
        self.open_age_hours
            .extend_from_slice(&other.open_age_hours);
        self.prediction_error
            .extend_from_slice(&other.prediction_error);
        self.prediction_abs_error
            .extend_from_slice(&other.prediction_abs_error);
    }

    fn profile(&self) -> LearnedProfile {
        let avg_profit = robust_weighted_mean(&self.profit, 0.0);
        let avg_roi = robust_weighted_mean(&self.roi, 0.0);
        let avg_hold_hours =
            robust_weighted_mean(&self.hold_hours, 0.0);
        let avg_sale_price =
            robust_weighted_mean(&self.sale_price, 0.0);
        let avg_buy_fill_hours =
            robust_weighted_mean(&self.buy_fill_hours, 0.0);
        let avg_sell_fill_hours =
            robust_weighted_mean(&self.sell_fill_hours, 0.0);
        let avg_open_age_hours =
            robust_weighted_mean(&self.open_age_hours, 0.0);

        let prediction_bias =
            robust_weighted_mean(&self.prediction_error, 0.0);
        let prediction_mae =
            robust_weighted_mean(&self.prediction_abs_error, 0.0);

        let prediction_accuracy = if self.prediction_abs_error.is_empty() {
            0.50
        } else {
            (1.0 / (1.0 + prediction_mae / (avg_profit.abs() + 5.0)))
                .clamp(0.10, 1.0)
        };

        let mad = weighted_mad(&self.profit).unwrap_or(0.0);
        let profit_stability = if self.profit.is_empty() {
            0.50
        } else {
            (1.0 / (1.0 + mad / (avg_profit.abs() + 5.0)))
                .clamp(0.15, 1.0)
        };

        let sample_weight = total_weight(&self.profit);
        let sample_confidence =
            sample_weight / (sample_weight + CONFIDENCE_PRIOR_UNITS);
        let trade_confidence =
            self.trade_count as f64 / (self.trade_count as f64 + 8.0);

        let confidence =
            (sample_confidence * 0.85 + trade_confidence * 0.15)
                .clamp(0.0, 1.0);

        let sell_through = if self.purchase_units > 0 {
            (self.matched_units as f64 / self.purchase_units as f64)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        LearnedProfile {
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

#[derive(Clone, Copy, Debug)]
struct RobustLevel {
    price: i64,
    cluster_median: i64,
    removed: usize,
}

fn robust_floor(
    prices: &[i64],
    historical_reference: f64,
) -> Option<RobustLevel> {
    let mut prices: Vec<i64> =
        prices.iter().copied().filter(|price| *price > 0).collect();

    if prices.is_empty() {
        return None;
    }

    prices.sort_unstable();

    let mut start = 0usize;
    let mut removed = 0usize;

    while start + 2 < prices.len() && removed < 2 {
        let current = prices[start];
        let next = prices[start + 1];
        let gap = next - current;
        let ratio = gap as f64 / current.max(1) as f64;
        let duplicate = prices[start + 1] == current;

        let historical_supports_next = historical_reference <= 0.0
            || (next as f64 - historical_reference).abs()
                <= (current as f64 - historical_reference).abs();

        if !duplicate
            && gap >= 3
            && ratio >= 0.15
            && historical_supports_next
        {
            start += 1;
            removed += 1;
        } else {
            break;
        }
    }

    let cluster = &prices[start..];
    let sample = &cluster[..cluster.len().min(5)];

    Some(RobustLevel {
        price: cluster[0],
        cluster_median: integer_median(sample),
        removed,
    })
}

fn robust_ceiling(prices: &[i64]) -> Option<RobustLevel> {
    let mut prices: Vec<i64> =
        prices.iter().copied().filter(|price| *price > 0).collect();

    if prices.is_empty() {
        return None;
    }

    prices.sort_unstable();

    let mut end = prices.len();
    let mut removed = 0usize;

    while end >= 3 && removed < 2 {
        let current = prices[end - 1];
        let next = prices[end - 2];
        let gap = current - next;
        let ratio = gap as f64 / next.max(1) as f64;
        let duplicate = prices[end - 2] == current;

        if !duplicate && gap >= 3 && ratio >= 0.15 {
            end -= 1;
            removed += 1;
        } else {
            break;
        }
    }

    let cluster = &prices[..end];
    let start = cluster.len().saturating_sub(5);
    let sample = &cluster[start..];

    Some(RobustLevel {
        price: *cluster.last().unwrap_or(&1),
        cluster_median: integer_median(sample),
        removed,
    })
}

fn build_bandit(
    store: &DecisionStore,
) -> HashMap<(TradeSide, String, String), ContextBanditStats> {
    let mut result: HashMap<
        (TradeSide, String, String),
        ContextBanditStats,
    > = HashMap::new();

    for decision in &store.decisions {
        if decision.model_version != MODEL_VERSION {
            continue;
        }

        let key = (
            decision.side,
            decision.context_key.clone(),
            decision.action.clone(),
        );

        let stats = result.entry(key).or_default();

        let age_or_fill = decision
            .actual_fill_hours
            .unwrap_or_else(|| {
                (decision.updated_at - decision.created_at)
                    .num_seconds()
                    .max(0) as f64
                    / 3600.0
            })
            .max(0.05);

        stats.exposure_hours += age_or_fill;

        match decision.status {
            DecisionStatus::Filled | DecisionStatus::Completed => {
                stats.successes += 1.0;
                stats.trials += 1.0;
            }
            DecisionStatus::Expired => {
                stats.failures += 1.0;
                stats.trials += 1.0;
            }
            DecisionStatus::Open => {}
        }

        if let Some(reward) = decision.actual_reward {
            stats.reward_samples.push(WeightedSample {
                value: reward,
                weight: recency_weight(decision.updated_at),
            });
        }
    }

    result
}

fn reconcile_decision_outcomes(
    store: &mut DecisionStore,
    transactions: &[TransactionModel],
    active_decision_ids: &HashSet<String>,
) {
    let now = Utc::now();

    // First: direct fill/sale events by decision id copied into transactions.
    for transaction in transactions {
        let Some(decision_id) =
            transaction_property_string(transaction, "algoframe_decision_id")
        else {
            continue;
        };

        let Some(decision) = store.find_mut(&decision_id) else {
            continue;
        };

        let fill_hours =
            (transaction.created_at - decision.created_at)
                .num_seconds()
                .max(0) as f64
                / 3600.0;

        decision.filled_at = Some(transaction.created_at);
        decision.actual_fill_hours = Some(fill_hours);
        decision.updated_at = transaction.created_at;

        if decision.side == TradeSide::Sell {
            decision.status = DecisionStatus::Completed;
            decision.completed_at = Some(transaction.created_at);

            let actual_profit = transaction
                .profit
                .map(|value| {
                    value as f64 / transaction.quantity.max(1) as f64
                })
                .unwrap_or_else(|| {
                    let bought: i64 = transaction_property_i64(
                        transaction,
                        "algoframe_bought_price",
                    )
                    .unwrap_or(decision.price);

                    transaction.price as f64
                        / transaction.quantity.max(1) as f64
                        - bought as f64
                });

            decision.actual_profit = Some(actual_profit);

            let reward = normalized_reward(
                actual_profit * transaction.quantity.max(1) as f64,
                decision.capital.max(1.0),
                fill_hours.max(0.10),
            );

            decision.actual_cycle_hours = Some(fill_hours);
            decision.actual_reward = Some(reward);
        } else {
            decision.status = DecisionStatus::Filled;
        }
    }

    // Second: FIFO-match purchases to later sales. This gives WTB actions their
    // true end-to-end reward rather than rewarding only a fast purchase.
    let mut lots_by_item: HashMap<String, VecDeque<PurchaseLot>> =
        HashMap::new();

    #[derive(Default)]
    struct BuyOutcome {
        units: i64,
        weighted_profit: f64,
        weighted_cycle_hours: f64,
        latest: Option<DateTime<Utc>>,
    }

    let mut buy_outcomes: HashMap<String, BuyOutcome> = HashMap::new();

    for transaction in transactions {
        if transaction.quantity <= 0 || transaction.price <= 0 {
            continue;
        }

        let key = item_key(&transaction.wfm_id, &transaction.sub_type);
        let unit_price =
            transaction.price as f64 / transaction.quantity as f64;

        if transaction.transaction_type == TransactionType::Purchase {
            lots_by_item
                .entry(key)
                .or_default()
                .push_back(PurchaseLot {
                    remaining: transaction.quantity,
                    unit_price,
                    at: transaction.created_at,
                    decision_id: transaction_property_string(
                        transaction,
                        "algoframe_decision_id",
                    ),
                    predicted_profit: transaction_property_f64(
                        transaction,
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
                    (transaction.created_at - lot.at)
                        .num_seconds()
                        .max(0) as f64
                        / 3600.0;

                let outcome =
                    buy_outcomes.entry(decision_id.clone()).or_default();

                outcome.units += matched;
                outcome.weighted_profit += profit * matched as f64;
                outcome.weighted_cycle_hours +=
                    cycle_hours * matched as f64;
                outcome.latest = Some(transaction.created_at);
            }

            lot.remaining -= matched;
            remaining -= matched;

            if lot.remaining > 0 {
                lots.push_front(lot);
            }
        }
    }

    for (decision_id, outcome) in buy_outcomes {
        if outcome.units <= 0 {
            continue;
        }

        let Some(decision) = store.find_mut(&decision_id) else {
            continue;
        };

        let actual_profit =
            outcome.weighted_profit / outcome.units as f64;
        let cycle_hours =
            outcome.weighted_cycle_hours / outcome.units as f64;

        decision.status = DecisionStatus::Completed;
        decision.completed_at = outcome.latest;
        decision.updated_at = outcome.latest.unwrap_or(now);
        decision.actual_profit = Some(actual_profit);
        decision.actual_cycle_hours = Some(cycle_hours);
        decision.actual_reward = Some(normalized_reward(
            actual_profit * outcome.units as f64,
            decision.capital.max(1.0),
            cycle_hours.max(0.10),
        ));
    }

    // Third: learn from failed/superseded orders too. An open decision that is
    // no longer represented by an active order and survived long enough to be
    // meaningful is treated as a censored failure rather than disappearing.
    for decision in &mut store.decisions {
        if decision.status != DecisionStatus::Open {
            continue;
        }

        if active_decision_ids.contains(&decision.id) {
            decision.updated_at = now;
            continue;
        }

        let age_hours =
            (now - decision.created_at).num_seconds().max(0) as f64
                / 3600.0;

        if age_hours >= EXPIRED_ORDER_GRACE_HOURS {
            decision.status = DecisionStatus::Expired;
            decision.updated_at = now;
            decision.actual_cycle_hours = Some(age_hours);
            decision.actual_reward = Some(-0.02);
        }
    }
}

fn sample_bandit_utility(
    stats: &ContextBanditStats,
    prior_success: (f64, f64),
    predicted_reward: f64,
) -> f64 {
    let mut rng = thread_rng();

    let alpha = prior_success.0 + stats.successes.max(0.0);
    let beta = prior_success.1 + stats.failures.max(0.0);

    let sampled_fill = Beta::new(alpha.max(0.1), beta.max(0.1))
        .ok()
        .map(|distribution| distribution.sample(&mut rng))
        .unwrap_or(0.5);

    let learned_weight =
        (stats.trials / (stats.trials + 8.0)).clamp(0.0, 0.85);

    let sampled_learned_reward = if stats.reward_samples.is_empty() {
        predicted_reward
    } else {
        let mean = stats.reward_mean();
        let std = stats.reward_std()
            / (stats.reward_samples.len() as f64).sqrt().max(1.0);

        Normal::new(mean, std.max(0.02))
            .ok()
            .map(|distribution| distribution.sample(&mut rng))
            .unwrap_or(mean)
    };

    let reward = predicted_reward * (1.0 - learned_weight)
        + sampled_learned_reward * learned_weight;

    // Thompson-sampled fill probability provides exploration automatically.
    sampled_fill * reward
}

fn action_prior_success(action: BuyAction) -> (f64, f64) {
    match action {
        BuyAction::Match => (2.8, 2.2),
        BuyAction::PlusOne => (3.0, 2.0),
        BuyAction::PlusTwo => (2.5, 2.5),
    }
}

fn sell_action_prior_success(action: SellAction) -> (f64, f64) {
    match action {
        SellAction::Hold => (2.3, 2.7),
        SellAction::MatchFloor => (3.0, 2.0),
        SellAction::UndercutOne => (3.2, 1.8),
        SellAction::UndercutTwo => (3.3, 1.7),
    }
}

fn make_context_key(
    side: TradeSide,
    tags: &[String],
    liquidity: f64,
    spread: f64,
    volatility: f64,
    pressure: f64,
    regime: MarketRegime,
) -> String {
    let category = tags
        .iter()
        .map(normalize_tag)
        .find(|tag| !tag.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    format!(
        "{}|{}|liq:{}|spr:{}|vol:{}|inv:{}|reg:{}",
        match side {
            TradeSide::Buy => "buy",
            TradeSide::Sell => "sell",
        },
        category,
        bucket(liquidity, &[0.20, 0.45, 0.70]),
        bucket(spread, &[0.08, 0.18, 0.35]),
        bucket(volatility, &[0.15, 0.35, 0.70]),
        bucket(pressure, &[0.20, 0.45, 0.70]),
        regime.as_str(),
    )
}

fn bucket(value: f64, cuts: &[f64]) -> usize {
    cuts.iter()
        .position(|cut| value < *cut)
        .unwrap_or(cuts.len())
}

fn detect_regime(
    live_reference: f64,
    historical_reference: f64,
    week_shift: f64,
) -> MarketRegime {
    if live_reference <= 0.0 || historical_reference <= 0.0 {
        return MarketRegime::Normal;
    }

    let displacement =
        ((live_reference - historical_reference) / historical_reference)
            .abs();

    let weekly = (week_shift / 100.0).abs();
    let score = displacement.max(weekly);

    if score >= 0.28 {
        MarketRegime::Shock
    } else if score >= 0.12 {
        MarketRegime::Moving
    } else {
        MarketRegime::Normal
    }
}

fn normalized_reward(
    profit: f64,
    capital: f64,
    hours: f64,
) -> f64 {
    if capital <= 0.0 || hours <= 0.0 {
        return 0.0;
    }

    // Percentage return per hour. This makes the bandit learn capital
    // efficiency instead of raw platinum alone.
    (profit / capital) / hours * 100.0
}

fn inventory_pressure(profile: &LearnedProfile) -> f64 {
    if profile.purchase_units <= 0 || profile.open_units <= 0 {
        return 0.0;
    }

    let open_ratio =
        profile.open_units as f64 / profile.purchase_units.max(1) as f64;

    let age_multiplier =
        1.0 + (profile.avg_open_age_hours / 168.0).clamp(0.0, 2.0);

    (open_ratio * age_multiplier).clamp(0.0, 1.0)
}

fn liquidity_score(volume: f64) -> f64 {
    (1.0 - (-volume.max(0.0) / 20.0).exp()).clamp(0.0, 1.0)
}

fn price_volatility(price: &ItemPriceInfo) -> f64 {
    let denominator = price
        .median
        .max(price.avg_price)
        .max(price.moving_avg.unwrap_or(0.0))
        .max(1.0);

    ((price.max_price - price.min_price).abs() / denominator)
        .clamp(0.0, 4.0)
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

fn normalize_disabled(value: i64) -> i64 {
    if value <= -1 {
        0
    } else {
        value.max(0)
    }
}

fn order_started_at(properties: &WfmProperties) -> Option<DateTime<Utc>> {
    let value: String = properties.get_property_value(
        "algoframe_order_started_at",
        String::new(),
    );

    if value.is_empty() {
        return None;
    }

    DateTime::parse_from_rfc3339(&value)
        .ok()
        .map(|value| value.with_timezone(&Utc))
}

pub fn order_age_hours(properties: &WfmProperties) -> f64 {
    order_started_at(properties)
        .map(|started| {
            (Utc::now() - started).num_seconds().max(0) as f64 / 3600.0
        })
        .unwrap_or(0.0)
}

fn ensure_order_started_at(properties: &mut WfmProperties) {
    if order_started_at(properties).is_none() {
        properties.set_property_value(
            "algoframe_order_started_at",
            Utc::now().to_rfc3339(),
        );
    }
}

fn transaction_order_age_hours(
    transaction: &TransactionModel,
) -> Option<f64> {
    let started = transaction_property_string(
        transaction,
        "algoframe_order_started_at",
    )?;

    let started = DateTime::parse_from_rfc3339(&started)
        .ok()?
        .with_timezone(&Utc);

    Some(
        (transaction.created_at - started)
            .num_seconds()
            .max(0) as f64
            / 3600.0,
    )
}

fn transaction_property_f64(
    transaction: &TransactionModel,
    key: &str,
) -> Option<f64> {
    let value = transaction.properties.as_ref()?.get(key)?;

    value
        .as_f64()
        .or_else(|| value.as_i64().map(|value| value as f64))
        .or_else(|| value.as_u64().map(|value| value as f64))
}

fn transaction_property_i64(
    transaction: &TransactionModel,
    key: &str,
) -> Option<i64> {
    let value = transaction.properties.as_ref()?.get(key)?;
    value
        .as_i64()
        .or_else(|| value.as_u64().map(|value| value as i64))
}

fn transaction_property_string(
    transaction: &TransactionModel,
    key: &str,
) -> Option<String> {
    transaction
        .properties
        .as_ref()?
        .get(key)?
        .as_str()
        .map(str::to_string)
}

fn collect_tags(
    transactions: &[TransactionModel],
) -> HashSet<String> {
    let mut result = HashSet::new();

    for transaction in transactions {
        for tag in transaction.tags.split(',') {
            let tag = normalize_tag(tag);
            if !tag.is_empty() {
                result.insert(tag);
            }
        }
    }

    result
}

fn normalize_tag(tag: impl AsRef<str>) -> String {
    tag.as_ref().trim().to_lowercase()
}

fn item_key(
    wfm_id: &str,
    sub_type: &Option<SubType>,
) -> String {
    let subtype =
        serde_json::to_string(sub_type).unwrap_or_else(|_| "null".to_string());

    format!("{wfm_id}|{subtype}")
}

fn recency_weight(at: DateTime<Utc>) -> f64 {
    let age_days =
        (Utc::now() - at).num_seconds().max(0) as f64 / 86_400.0;

    0.5_f64.powf(age_days / OUTCOME_HALF_LIFE_DAYS)
}

fn robust_weighted_mean(
    samples: &[WeightedSample],
    fallback: f64,
) -> f64 {
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

        let value =
            sample.value.clamp(median - clip, median + clip);

        weighted_sum += value * sample.weight;
        total += sample.weight;
    }

    if total > 0.0 {
        weighted_sum / total
    } else {
        fallback
    }
}

fn weighted_median(samples: &[WeightedSample]) -> Option<f64> {
    let mut samples: Vec<WeightedSample> = samples
        .iter()
        .copied()
        .filter(|sample| sample.weight > 0.0 && sample.value.is_finite())
        .collect();

    if samples.is_empty() {
        return None;
    }

    samples.sort_by(|a, b| {
        a.value
            .partial_cmp(&b.value)
            .unwrap_or(Ordering::Equal)
    });

    let total = samples.iter().map(|sample| sample.weight).sum::<f64>();
    let midpoint = total / 2.0;
    let mut current = 0.0;

    for sample in samples {
        current += sample.weight;
        if current >= midpoint {
            return Some(sample.value);
        }
    }

    None
}

fn weighted_mad(samples: &[WeightedSample]) -> Option<f64> {
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

fn total_weight(samples: &[WeightedSample]) -> f64 {
    samples
        .iter()
        .filter(|sample| sample.weight > 0.0)
        .map(|sample| sample.weight)
        .sum()
}

fn integer_median(values: &[i64]) -> i64 {
    if values.is_empty() {
        return 0;
    }

    let mut values = values.to_vec();
    values.sort_unstable();
    values[values.len() / 2]
}

fn blend_profiles(
    base: &LearnedProfile,
    newer: &LearnedProfile,
    weight: f64,
) -> LearnedProfile {
    let weight = weight.clamp(0.0, 1.0);
    let inverse = 1.0 - weight;
    let mix = |a: f64, b: f64| a * inverse + b * weight;

    LearnedProfile {
        purchase_units: newer.purchase_units,
        sold_units: newer.sold_units,
        matched_units: newer.matched_units,
        open_units: newer.open_units,
        trade_count: newer.trade_count,

        avg_profit: mix(base.avg_profit, newer.avg_profit),
        avg_roi: mix(base.avg_roi, newer.avg_roi),
        avg_hold_hours: mix(base.avg_hold_hours, newer.avg_hold_hours),
        avg_sale_price: mix(base.avg_sale_price, newer.avg_sale_price),
        avg_buy_fill_hours: mix(
            base.avg_buy_fill_hours,
            newer.avg_buy_fill_hours,
        ),
        avg_sell_fill_hours: mix(
            base.avg_sell_fill_hours,
            newer.avg_sell_fill_hours,
        ),
        avg_open_age_hours: newer.avg_open_age_hours,

        prediction_bias: mix(
            base.prediction_bias,
            newer.prediction_bias,
        ),
        prediction_mae: mix(
            base.prediction_mae,
            newer.prediction_mae,
        ),
        prediction_accuracy: mix(
            base.prediction_accuracy,
            newer.prediction_accuracy,
        ),

        sell_through: mix(
            base.sell_through,
            newer.sell_through,
        ),
        profit_stability: mix(
            base.profit_stability,
            newer.profit_stability,
        ),

        confidence: newer.confidence,
    }
}

fn average_profiles(
    profiles: &[LearnedProfile],
) -> Option<LearnedProfile> {
    if profiles.is_empty() {
        return None;
    }

    let weights: Vec<f64> = profiles
        .iter()
        .map(|profile| 0.10 + profile.confidence)
        .collect();

    let total = weights.iter().sum::<f64>().max(f64::EPSILON);

    let average = |selector: fn(&LearnedProfile) -> f64| {
        profiles
            .iter()
            .zip(weights.iter())
            .map(|(profile, weight)| selector(profile) * weight)
            .sum::<f64>()
            / total
    };

    Some(LearnedProfile {
        avg_profit: average(|profile| profile.avg_profit),
        avg_roi: average(|profile| profile.avg_roi),
        avg_hold_hours: average(|profile| profile.avg_hold_hours),
        avg_sale_price: average(|profile| profile.avg_sale_price),
        avg_buy_fill_hours: average(|profile| profile.avg_buy_fill_hours),
        avg_sell_fill_hours: average(|profile| profile.avg_sell_fill_hours),
        avg_open_age_hours: average(|profile| profile.avg_open_age_hours),
        prediction_bias: average(|profile| profile.prediction_bias),
        prediction_mae: average(|profile| profile.prediction_mae),
        prediction_accuracy: average(
            |profile| profile.prediction_accuracy,
        ),
        sell_through: average(|profile| profile.sell_through),
        profit_stability: average(
            |profile| profile.profit_stability,
        ),
        confidence: average(|profile| profile.confidence),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_isolated_sell_dump() {
        let level =
            robust_floor(&[10, 20, 21, 22, 23], 21.0).unwrap();

        assert_eq!(level.price, 20);
        assert_eq!(level.removed, 1);
    }

    #[test]
    fn keeps_repeated_low_sell_cluster() {
        let level =
            robust_floor(&[10, 10, 20, 21, 22], 21.0).unwrap();

        assert_eq!(level.price, 10);
        assert_eq!(level.removed, 0);
    }

    #[test]
    fn removes_isolated_high_buy_spike() {
        let level = robust_ceiling(&[20, 21, 22, 40]).unwrap();

        assert_eq!(level.price, 22);
        assert_eq!(level.removed, 1);
    }

    #[test]
    fn detects_market_shock() {
        assert_eq!(
            detect_regime(140.0, 100.0, 0.0),
            MarketRegime::Shock
        );
    }

    #[test]
    fn reward_prefers_fast_capital_turnover() {
        let fast = normalized_reward(15.0, 50.0, 1.0);
        let slow = normalized_reward(30.0, 100.0, 24.0);

        assert!(fast > slow);
    }

    #[test]
    fn context_changes_with_regime() {
        let normal = make_context_key(
            TradeSide::Buy,
            &["prime".to_string()],
            0.5,
            0.2,
            0.2,
            0.1,
            MarketRegime::Normal,
        );

        let shock = make_context_key(
            TradeSide::Buy,
            &["prime".to_string()],
            0.5,
            0.2,
            0.2,
            0.1,
            MarketRegime::Shock,
        );

        assert_ne!(normal, shock);
    }
}
