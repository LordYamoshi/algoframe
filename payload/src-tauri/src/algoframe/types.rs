
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const MODEL_VERSION: &str = "ultimate_v5";
pub const FEATURE_SCHEMA_VERSION: u32 = 5;
pub const REWARD_VERSION: u32 = 5;
pub const POLICY_VERSION: u32 = 5;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperatingMode {
    Paper,
    Conservative,
    Balanced,
    Growth,
    Liquid,
}

impl Default for OperatingMode {
    fn default() -> Self {
        Self::Balanced
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ItemLifecycle {
    Standard,
    NewRelease,
    Mature,
    RecentlyVaulted,
    Resurgence,
    EventAffected,
}

impl Default for ItemLifecycle {
    fn default() -> Self {
        Self::Standard
    }
}

impl ItemLifecycle {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::NewRelease => "new_release",
            Self::Mature => "mature",
            Self::RecentlyVaulted => "recently_vaulted",
            Self::Resurgence => "resurgence",
            Self::EventAffected => "event_affected",
        }
    }

    pub fn numeric(self) -> f64 {
        match self {
            Self::Standard => 0.0,
            Self::NewRelease => 0.20,
            Self::Mature => 0.40,
            Self::RecentlyVaulted => 0.60,
            Self::Resurgence => 0.80,
            Self::EventAffected => 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MarketRegime {
    Normal,
    Moving,
    Shock,
}

impl Default for MarketRegime {
    fn default() -> Self {
        Self::Normal
    }
}

impl MarketRegime {
    pub fn confidence_multiplier(self) -> f64 {
        match self {
            Self::Normal => 1.0,
            Self::Moving => 0.55,
            Self::Shock => 0.25,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Moving => "moving",
            Self::Shock => "shock",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleState {
    Discovered,
    Observed,
    Candidate,
    Allocated,
    OrderActive,
    PartiallyFilled,
    Purchased,
    SellActive,
    Sold,
    Failed,
    Expired,
    Abandoned,
    PaperSimulated,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStatus {
    Open,
    Partial,
    Filled,
    Completed,
    Expired,
    Rejected,
    Cancelled,
    PaperOpen,
    PaperCompleted,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Champion,
    Challenger,
    Shadow,
    Fallback,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct UltimateConfig {
    pub enabled: bool,
    pub mode: OperatingMode,

    pub cash_reserve_pct: f64,
    pub max_item_exposure_pct: f64,
    pub max_category_exposure_pct: f64,
    pub max_family_exposure_pct: f64,
    pub max_high_volatility_exposure_pct: f64,
    pub experimental_budget_pct: f64,
    pub max_single_decision_pct: f64,
    pub max_trade_quantity: i64,

    pub max_daily_trade_interactions: i64,
    pub human_minutes_per_trade: f64,
    pub human_time_value_plat_per_hour: f64,

    pub min_reprice_minutes: f64,
    pub stable_reprice_minutes: f64,
    pub reprice_min_reward_gain_pct: f64,

    pub minimum_data_quality: f64,
    pub minimum_model_health: f64,
    pub max_drawdown_pct: f64,
    pub max_inventory_pressure: f64,

    pub recency_half_life_days: f64,
    pub snapshot_full_days: i64,
    pub snapshot_five_min_days: i64,
    pub snapshot_thirty_min_days: i64,
    pub snapshot_hourly_days: i64,

    pub max_price_search_steps: i64,
    pub max_quantity_search: i64,
    pub propensity_samples: usize,
    pub nearest_neighbor_count: usize,

    pub champion_policy: String,
    pub challenger_policy: String,
    pub challenger_fraction: f64,
    pub minimum_promotion_samples: usize,
    pub promotion_margin_pct: f64,
    pub maximum_challenger_failure_delta: f64,

    pub automatic_tuning: bool,
    pub automatic_promotion: bool,
    pub automatic_rollback: bool,
    pub shadow_mode: bool,
    pub market_recording: bool,
    pub event_intelligence: bool,
    pub arbitrage_engine: bool,
    pub continuous_price_optimization: bool,
    pub quantity_optimization: bool,
    pub survival_models: bool,
    pub hierarchical_learning: bool,
    pub anomaly_detection: bool,
    pub regime_detection: bool,
    pub opportunity_forecasting: bool,

    // Ultimate V5: distributional / causal / advanced risk intelligence.
    pub distributional_predictions: bool,
    pub cvar_risk_optimization: bool,
    pub conformal_intervals: bool,
    pub uncertainty_decomposition: bool,
    pub learned_market_states: bool,
    pub book_persistence_model: bool,
    pub causal_evaluation: bool,
    pub active_learning: bool,
    pub pareto_portfolio: bool,
    pub walk_forward_validation: bool,
    pub stress_testing: bool,
    pub monte_carlo_portfolio: bool,
    pub automatic_ablation: bool,
    pub leakage_detection: bool,
    pub nonlinear_expert: bool,
    pub mixture_of_experts: bool,

    pub cvar_alpha: f64,
    pub downside_risk_weight: f64,
    pub epistemic_risk_weight: f64,
    pub aleatoric_risk_weight: f64,
    pub active_learning_budget_pct: f64,

    pub settings_revision: u64,
}

impl Default for UltimateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: OperatingMode::Balanced,

            cash_reserve_pct: 0.15,
            max_item_exposure_pct: 0.15,
            max_category_exposure_pct: 0.30,
            max_family_exposure_pct: 0.22,
            max_high_volatility_exposure_pct: 0.20,
            experimental_budget_pct: 0.05,
            max_single_decision_pct: 0.10,
            max_trade_quantity: 6,

            max_daily_trade_interactions: 80,
            human_minutes_per_trade: 1.5,
            human_time_value_plat_per_hour: 30.0,

            min_reprice_minutes: 4.0,
            stable_reprice_minutes: 12.0,
            reprice_min_reward_gain_pct: 0.08,

            minimum_data_quality: 0.50,
            minimum_model_health: 0.55,
            max_drawdown_pct: 0.15,
            max_inventory_pressure: 0.90,

            recency_half_life_days: 30.0,
            snapshot_full_days: 7,
            snapshot_five_min_days: 30,
            snapshot_thirty_min_days: 180,
            snapshot_hourly_days: 730,

            max_price_search_steps: 8,
            max_quantity_search: 6,
            propensity_samples: 96,
            nearest_neighbor_count: 24,

            champion_policy: "balanced".to_string(),
            challenger_policy: "fast_turnover".to_string(),
            challenger_fraction: 0.10,
            minimum_promotion_samples: 200,
            promotion_margin_pct: 0.05,
            maximum_challenger_failure_delta: 0.05,

            automatic_tuning: true,
            automatic_promotion: true,
            automatic_rollback: true,
            shadow_mode: true,
            market_recording: true,
            event_intelligence: true,
            arbitrage_engine: true,
            continuous_price_optimization: true,
            quantity_optimization: true,
            survival_models: true,
            hierarchical_learning: true,
            anomaly_detection: true,
            regime_detection: true,
            opportunity_forecasting: true,

            distributional_predictions: true,
            cvar_risk_optimization: true,
            conformal_intervals: true,
            uncertainty_decomposition: true,
            learned_market_states: true,
            book_persistence_model: true,
            causal_evaluation: true,
            active_learning: true,
            pareto_portfolio: true,
            walk_forward_validation: true,
            stress_testing: true,
            monte_carlo_portfolio: true,
            automatic_ablation: true,
            leakage_detection: true,
            nonlinear_expert: true,
            mixture_of_experts: true,

            cvar_alpha: 0.10,
            downside_risk_weight: 0.30,
            epistemic_risk_weight: 0.20,
            aleatoric_risk_weight: 0.20,
            active_learning_budget_pct: 0.03,

            settings_revision: 1,
        }
    }
}

impl UltimateConfig {
    pub fn normalized(mut self) -> Self {
        self.cash_reserve_pct = self.cash_reserve_pct.clamp(0.0, 0.80);
        self.max_item_exposure_pct = self.max_item_exposure_pct.clamp(0.01, 1.0);
        self.max_category_exposure_pct = self.max_category_exposure_pct.clamp(0.01, 1.0);
        self.max_family_exposure_pct = self.max_family_exposure_pct.clamp(0.01, 1.0);
        self.max_high_volatility_exposure_pct =
            self.max_high_volatility_exposure_pct.clamp(0.01, 1.0);
        self.experimental_budget_pct = self.experimental_budget_pct.clamp(0.0, 0.30);
        self.max_single_decision_pct = self.max_single_decision_pct.clamp(0.01, 1.0);
        self.max_trade_quantity = self.max_trade_quantity.clamp(1, 99);
        self.max_daily_trade_interactions = self.max_daily_trade_interactions.clamp(1, 500);
        self.human_minutes_per_trade = self.human_minutes_per_trade.clamp(0.0, 60.0);
        self.human_time_value_plat_per_hour =
            self.human_time_value_plat_per_hour.clamp(0.0, 10_000.0);
        self.min_reprice_minutes = self.min_reprice_minutes.clamp(0.0, 120.0);
        self.stable_reprice_minutes =
            self.stable_reprice_minutes.clamp(self.min_reprice_minutes, 240.0);
        self.reprice_min_reward_gain_pct =
            self.reprice_min_reward_gain_pct.clamp(0.0, 1.0);
        self.minimum_data_quality = self.minimum_data_quality.clamp(0.0, 1.0);
        self.minimum_model_health = self.minimum_model_health.clamp(0.0, 1.0);
        self.max_drawdown_pct = self.max_drawdown_pct.clamp(0.01, 1.0);
        self.max_inventory_pressure = self.max_inventory_pressure.clamp(0.05, 1.0);
        self.recency_half_life_days = self.recency_half_life_days.clamp(3.0, 180.0);
        self.propensity_samples = self.propensity_samples.clamp(24, 512);
        self.nearest_neighbor_count = self.nearest_neighbor_count.clamp(4, 128);
        const VALID_POLICIES: &[&str] = &[
            "balanced",
            "fast_turnover",
            "max_reward",
            "conservative",
            "aggressive",
        ];

        if !VALID_POLICIES.contains(&self.champion_policy.as_str()) {
            self.champion_policy = "balanced".to_string();
        }
        if !VALID_POLICIES.contains(&self.challenger_policy.as_str()) {
            self.challenger_policy = "fast_turnover".to_string();
        }

        self.challenger_fraction = self.challenger_fraction.clamp(0.0, 0.50);
        self.minimum_promotion_samples = self.minimum_promotion_samples.clamp(20, 100_000);
        self.promotion_margin_pct = self.promotion_margin_pct.clamp(0.0, 1.0);
        self.maximum_challenger_failure_delta =
            self.maximum_challenger_failure_delta.clamp(0.0, 0.50);

        self.cvar_alpha = self.cvar_alpha.clamp(0.01, 0.40);
        self.downside_risk_weight = self.downside_risk_weight.clamp(0.0, 1.0);
        self.epistemic_risk_weight = self.epistemic_risk_weight.clamp(0.0, 1.0);
        self.aleatoric_risk_weight = self.aleatoric_risk_weight.clamp(0.0, 1.0);
        self.active_learning_budget_pct =
            self.active_learning_budget_pct.clamp(0.0, 0.20);
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct OrderBookDynamics {
    pub bid_velocity_per_hour: f64,
    pub ask_velocity_per_hour: f64,
    pub spread_velocity_per_hour: f64,
    pub buy_depth_change_per_hour: f64,
    pub sell_depth_change_per_hour: f64,
    pub churn_score: f64,
    pub competitive_response_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Forecast {
    pub price_30m: f64,
    pub price_2h: f64,
    pub price_8h: f64,
    pub spread_30m: f64,
    pub opportunity_2h: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AnomalyReport {
    pub score: f64,
    pub suspicious_low_listings: usize,
    pub suspicious_high_bids: usize,
    pub sudden_price_jump: bool,
    pub suspicious_churn: bool,
    pub shallow_book: bool,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct DataQuality {
    pub score: f64,
    pub historical_quality: f64,
    pub order_book_quality: f64,
    pub depth_quality: f64,
    pub freshness_quality: f64,
    pub anomaly_penalty: f64,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MarketFeatures {
    pub best_bid: i64,
    pub best_ask: i64,
    pub robust_bid: i64,
    pub robust_ask: i64,
    pub mid_price: f64,
    pub spread: f64,
    pub spread_percent: f64,

    pub buy_depth: i64,
    pub sell_depth: i64,
    pub volume: f64,
    pub liquidity: f64,
    pub volatility: f64,
    pub week_shift: f64,

    pub inventory_pressure: f64,
    pub time_hour_sin: f64,
    pub time_hour_cos: f64,
    pub weekday_sin: f64,
    pub weekday_cos: f64,

    pub event_signal: f64,
    pub arbitrage_signal: f64,
    pub opportunity_cost_signal: f64,

    pub regime: MarketRegime,
    pub regime_score: f64,
    pub lifecycle: ItemLifecycle,
    pub anomaly: AnomalyReport,
    pub quality: DataQuality,
    pub dynamics: OrderBookDynamics,
    pub forecast: Forecast,

    pub embedding: Vec<f64>,
}

impl MarketFeatures {
    pub fn flat_map(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("best_bid".into(), self.best_bid as f64),
            ("best_ask".into(), self.best_ask as f64),
            ("mid_price".into(), self.mid_price),
            ("spread".into(), self.spread),
            ("spread_percent".into(), self.spread_percent),
            ("buy_depth".into(), self.buy_depth as f64),
            ("sell_depth".into(), self.sell_depth as f64),
            ("volume".into(), self.volume),
            ("liquidity".into(), self.liquidity),
            ("volatility".into(), self.volatility),
            ("week_shift".into(), self.week_shift),
            ("inventory_pressure".into(), self.inventory_pressure),
            ("event_signal".into(), self.event_signal),
            ("arbitrage_signal".into(), self.arbitrage_signal),
            ("opportunity_cost_signal".into(), self.opportunity_cost_signal),
            ("lifecycle".into(), self.lifecycle.numeric()),
            ("anomaly_score".into(), self.anomaly.score),
            ("data_quality".into(), self.quality.score),
            ("bid_velocity".into(), self.dynamics.bid_velocity_per_hour),
            ("ask_velocity".into(), self.dynamics.ask_velocity_per_hour),
            ("spread_velocity".into(), self.dynamics.spread_velocity_per_hour),
            ("churn".into(), self.dynamics.churn_score),
            ("competitive_response".into(), self.dynamics.competitive_response_score),
            ("forecast_30m".into(), self.forecast.price_30m),
            ("forecast_2h".into(), self.forecast.price_2h),
            ("forecast_8h".into(), self.forecast.price_8h),
            ("forecast_opportunity".into(), self.forecast.opportunity_2h),
        ])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketSnapshot {
    pub id: String,
    pub item_key: String,
    pub wfm_id: String,
    pub wfm_url: String,
    pub item_name: String,
    pub category: String,
    pub tags: Vec<String>,
    pub sub_type: serde_json::Value,

    pub created_at: DateTime<Utc>,
    pub external_only: bool,

    pub buy_prices: Vec<i64>,
    pub sell_prices: Vec<i64>,
    pub features: MarketFeatures,

    pub model_version: String,
    pub feature_schema_version: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SurvivalPrediction {
    pub fill_15m: f64,
    pub fill_1h: f64,
    pub fill_6h: f64,
    pub fill_24h: f64,
    pub median_fill_hours: f64,
    pub expected_fill_hours: f64,
    pub calibration: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RewardBreakdown {
    pub expected_profit: f64,
    pub expected_roi: f64,
    pub expected_platinum_per_hour: f64,
    pub capital_cost: f64,
    pub human_time_cost: f64,
    pub risk_penalty: f64,
    pub inventory_penalty: f64,
    pub anomaly_penalty: f64,
    pub opportunity_cost_penalty: f64,
    pub arbitrage_bonus: f64,
    pub event_adjustment: f64,
    pub final_reward: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Explanation {
    pub headline: String,
    pub positive_factors: Vec<String>,
    pub negative_factors: Vec<String>,
    pub guardrails: Vec<String>,
    pub confidence_notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateAction {
    pub key: String,
    pub side: TradeSide,
    pub price: i64,
    pub quantity: i64,
    pub price_offset: i64,
    pub expected_sell_price: f64,

    pub survival: SurvivalPrediction,
    pub reward: RewardBreakdown,

    pub bandit_sample: f64,
    pub propensity: f64,
    pub uncertainty: f64,

    pub valid: bool,
    pub reject_reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionIntent {
    pub allowed: bool,
    pub paper: bool,

    pub decision_id: String,
    pub snapshot_id: String,
    pub side: TradeSide,
    pub lifecycle: LifecycleState,

    pub wfm_id: String,
    pub item_key: String,
    pub item_name: String,
    pub category: String,

    pub price: i64,
    pub quantity: i64,
    pub capital: f64,

    pub expected_sell_price: f64,
    pub survival: SurvivalPrediction,
    pub reward: RewardBreakdown,
    pub confidence: f64,
    pub uncertainty: f64,

    pub context_key: String,
    pub selected_action: String,
    pub selected_propensity: f64,
    pub propensities: HashMap<String, f64>,
    pub shadow_actions: HashMap<String, String>,

    pub regime: MarketRegime,
    pub data_quality: f64,
    pub anomaly_score: f64,
    pub volatility: f64,
    pub model_role: ModelRole,
    pub policy_name: String,

    pub explanation: Explanation,

    pub seed: u64,
    pub model_version: String,
    pub feature_schema_version: u32,
    pub reward_version: u32,
    pub policy_version: u32,
    pub settings_hash: String,
}

impl ExecutionIntent {
    pub fn rejected(
        decision_id: String,
        snapshot_id: String,
        side: TradeSide,
        wfm_id: String,
        item_key: String,
        item_name: String,
        category: String,
        explanation: Explanation,
    ) -> Self {
        Self {
            allowed: false,
            paper: false,
            decision_id,
            snapshot_id,
            side,
            lifecycle: LifecycleState::Failed,
            wfm_id,
            item_key,
            item_name,
            category,
            price: 0,
            quantity: 0,
            capital: 0.0,
            expected_sell_price: 0.0,
            survival: SurvivalPrediction::default(),
            reward: RewardBreakdown::default(),
            confidence: 0.0,
            uncertainty: 1.0,
            context_key: String::new(),
            selected_action: "reject".to_string(),
            selected_propensity: 1.0,
            propensities: HashMap::new(),
            shadow_actions: HashMap::new(),
            regime: MarketRegime::Normal,
            data_quality: 0.0,
            anomaly_score: 0.0,
            volatility: 0.0,
            model_role: ModelRole::Fallback,
            policy_name: "conservative".to_string(),
            explanation,
            seed: 0,
            model_version: MODEL_VERSION.to_string(),
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            reward_version: REWARD_VERSION,
            policy_version: POLICY_VERSION,
            settings_hash: String::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub id: String,
    pub snapshot_id: String,

    pub wfm_id: String,
    pub item_key: String,
    pub item_name: String,
    pub category: String,

    pub side: TradeSide,
    pub lifecycle: LifecycleState,
    pub status: DecisionStatus,

    pub chosen_action: String,
    pub price: i64,
    pub quantity: i64,
    pub filled_quantity: i64,
    pub capital: f64,

    pub context_key: String,
    pub propensities: HashMap<String, f64>,
    pub chosen_propensity: f64,
    pub shadow_actions: HashMap<String, String>,

    pub predicted_profit: f64,
    pub predicted_reward: f64,
    pub predicted_fill: SurvivalPrediction,

    pub actual_profit: Option<f64>,
    pub actual_reward: Option<f64>,
    pub actual_fill_hours: Option<f64>,
    pub actual_cycle_hours: Option<f64>,
    pub fill_ratio: f64,

    pub features: MarketFeatures,
    pub reward_breakdown: RewardBreakdown,
    pub explanation: Explanation,

    pub regime: MarketRegime,
    pub model_role: ModelRole,
    pub policy_name: String,

    pub seed: u64,
    pub model_version: String,
    pub feature_schema_version: u32,
    pub reward_version: u32,
    pub policy_version: u32,
    pub settings_hash: String,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub filled_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutcomeRecord {
    pub id: String,
    pub decision_id: String,
    pub item_key: String,
    pub side: TradeSide,

    pub outcome_type: String,
    pub quantity: i64,
    pub profit: f64,
    pub reward: f64,
    pub fill_hours: f64,
    pub cycle_hours: f64,
    pub competition_response: f64,

    pub simulated: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelVersionRecord {
    pub id: String,
    pub version: String,
    pub role: ModelRole,
    pub active: bool,

    pub parameters: serde_json::Value,
    pub trained_on_decisions: usize,

    pub average_reward: f64,
    pub failure_rate: f64,
    pub prediction_mae: f64,
    pub calibration_error: f64,
    pub drawdown: f64,

    pub created_at: DateTime<Utc>,
    pub promoted_at: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EvaluationMetrics {
    pub sample_count: usize,
    pub average_reward: f64,
    pub median_reward: f64,
    pub failure_rate: f64,
    pub prediction_mae: f64,
    pub fill_brier_score: f64,
    pub calibration_error: f64,
    pub ips_reward: f64,
    pub snips_reward: f64,
    pub doubly_robust_reward: f64,
    pub max_drawdown: f64,
    pub profit_per_hour: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventSignal {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub keywords: Vec<String>,
    pub tags: Vec<String>,
    pub impact: f64,
    pub confidence: f64,
    pub starts_at: DateTime<Utc>,
    pub ends_at: Option<DateTime<Utc>>,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub from_key: String,
    pub to_key: String,
    pub relation: String,
    pub quantity: f64,
    pub cost: f64,
    pub metadata: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ArbitrageOpportunity {
    pub id: String,
    pub kind: String,
    pub family: String,
    pub legs: Vec<String>,
    pub cost: f64,
    pub expected_revenue: f64,
    pub expected_profit: f64,
    pub expected_hours: f64,
    pub score: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlertRecord {
    pub id: String,
    pub severity: String,
    pub code: String,
    pub message: String,
    pub item_key: Option<String>,
    pub created_at: DateTime<Utc>,
    pub acknowledged: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct FeatureImportance {
    pub feature: String,
    pub correlation: f64,
    pub importance: f64,
    pub sample_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ProfitAttribution {
    pub market_selection: f64,
    pub buy_price_optimization: f64,
    pub sell_price_optimization: f64,
    pub portfolio_allocation: f64,
    pub arbitrage: f64,
    pub event_intelligence: f64,
    pub exploration_cost: f64,
    pub human_time_cost: f64,
    pub total_realized: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelHealth {
    pub score: f64,
    pub healthy: bool,
    pub fallback_active: bool,
    pub reasons: Vec<String>,

    pub recent_reward: f64,
    pub baseline_reward: f64,
    pub recent_failure_rate: f64,
    pub recent_prediction_mae: f64,
    pub calibration_error: f64,
    pub drift_score: f64,
    pub inventory_backlog: f64,
    pub drawdown: f64,
}

impl Default for ModelHealth {
    fn default() -> Self {
        Self {
            score: 1.0,
            healthy: true,
            fallback_active: false,
            reasons: vec![],
            recent_reward: 0.0,
            baseline_reward: 0.0,
            recent_failure_rate: 0.0,
            recent_prediction_mae: 0.0,
            calibration_error: 0.0,
            drift_score: 0.0,
            inventory_backlog: 0.0,
            drawdown: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PortfolioState {
    pub total_capital_budget: f64,
    pub reserved_capital: f64,
    pub allocated_capital: f64,
    pub high_volatility_capital: f64,
    pub experimental_capital: f64,
    pub item_exposure: HashMap<String, f64>,
    pub family_exposure: HashMap<String, f64>,
    pub category_exposure: HashMap<String, f64>,
    pub estimated_daily_interactions: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PortfolioDecision {
    pub selected_order_ids: Vec<String>,
    pub rejected_order_ids: Vec<String>,
    pub reasons: HashMap<String, String>,
    pub state: PortfolioState,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct OfflinePolicyEvaluation {
    pub policy_name: String,
    pub metrics: EvaluationMetrics,
    pub confidence_low: f64,
    pub confidence_high: f64,
    pub promotable: bool,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct LearningInspector {
    pub model: String,
    pub feature_schema_version: u32,
    pub reward_version: u32,
    pub policy_version: u32,
    pub updated_at: Option<DateTime<Utc>>,

    pub mode: OperatingMode,
    pub health: ModelHealth,

    pub transactions_learned: usize,
    pub snapshots_recorded: usize,
    pub decisions_total: usize,
    pub decisions_completed: usize,
    pub decisions_failed: usize,
    pub decisions_open: usize,

    pub champion: Option<ModelVersionRecord>,
    pub challengers: Vec<ModelVersionRecord>,
    pub shadow_evaluations: Vec<OfflinePolicyEvaluation>,

    pub metrics: EvaluationMetrics,
    pub profit_attribution: ProfitAttribution,

    pub feature_importance: Vec<FeatureImportance>,
    pub arbitrage_opportunities: Vec<ArbitrageOpportunity>,
    pub alerts: Vec<AlertRecord>,
    pub active_events: Vec<EventSignal>,

    pub top_items: Vec<serde_json::Value>,
    pub portfolio: PortfolioState,

    #[serde(default)]
    pub advanced: serde_json::Value,

    pub config: UltimateConfig,
}

impl Default for LearningInspector {
    fn default() -> Self {
        Self {
            model: MODEL_VERSION.to_string(),
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            reward_version: REWARD_VERSION,
            policy_version: POLICY_VERSION,
            updated_at: None,
            mode: OperatingMode::Balanced,
            health: ModelHealth::default(),
            transactions_learned: 0,
            snapshots_recorded: 0,
            decisions_total: 0,
            decisions_completed: 0,
            decisions_failed: 0,
            decisions_open: 0,
            champion: None,
            challengers: vec![],
            shadow_evaluations: vec![],
            metrics: EvaluationMetrics::default(),
            profit_attribution: ProfitAttribution::default(),
            feature_importance: vec![],
            arbitrage_opportunities: vec![],
            alerts: vec![],
            active_events: vec![],
            top_items: vec![],
            portfolio: PortfolioState::default(),
            advanced: serde_json::Value::Null,
            config: UltimateConfig::default(),
        }
    }
}
