use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use service::sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, Statement, TryGetable,
};
use utils::{get_location, Error};
use wf_market::types::{Order, OrderList};

use super::{
    advanced::{leakage_report, synthetic_stress_tests, walk_forward_validation},
    math::offline_policy_evaluation,
    types::{
        DecisionRecord, DecisionStatus, ExecutionIntent, MarketSnapshot, ModelHealth,
        OfflinePolicyEvaluation, TradeSide,
    },
    LearningStore,
};

fn rel_error(component: &str, error: impl ToString) -> Error {
    Error::new(
        format!("AlgoFrame:Reliability:{component}"),
        error.to_string(),
        get_location!(),
    )
}

fn sql_string(value: &str) -> String {
    value.replace('\'', "''")
}

async fn execute(
    db: &DatabaseConnection,
    component: &str,
    sql: impl Into<String>,
) -> Result<(), Error> {
    db.execute(Statement::from_string(DatabaseBackend::Sqlite, sql.into()))
        .await
        .map_err(|error| rel_error(component, error))?;
    Ok(())
}

fn side_string(side: TradeSide) -> &'static str {
    match side {
        TradeSide::Buy => "buy",
        TradeSide::Sell => "sell",
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReliabilityLifecycleState {
    Discovered,
    Evaluated,
    Approved,
    Prepared,
    Dispatching,
    Active,
    Partial,
    Purchased,
    Selling,
    Sold,
    PaperSimulated,
    Rejected,
    Cancelled,
    Superseded,
    Expired,
    Failed,
    Unknown,
}

impl ReliabilityLifecycleState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Discovered => "discovered",
            Self::Evaluated => "evaluated",
            Self::Approved => "approved",
            Self::Prepared => "prepared",
            Self::Dispatching => "dispatching",
            Self::Active => "active",
            Self::Partial => "partial",
            Self::Purchased => "purchased",
            Self::Selling => "selling",
            Self::Sold => "sold",
            Self::PaperSimulated => "paper_simulated",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
            Self::Superseded => "superseded",
            Self::Expired => "expired",
            Self::Failed => "failed",
            Self::Unknown => "unknown",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "discovered" => Self::Discovered,
            "evaluated" => Self::Evaluated,
            "approved" => Self::Approved,
            "prepared" => Self::Prepared,
            "dispatching" => Self::Dispatching,
            "active" => Self::Active,
            "partial" => Self::Partial,
            "purchased" => Self::Purchased,
            "selling" => Self::Selling,
            "sold" => Self::Sold,
            "paper_simulated" => Self::PaperSimulated,
            "rejected" => Self::Rejected,
            "cancelled" => Self::Cancelled,
            "superseded" => Self::Superseded,
            "expired" => Self::Expired,
            "failed" => Self::Failed,
            _ => Self::Unknown,
        }
    }

    fn rank(self) -> i32 {
        match self {
            Self::Discovered => 0,
            Self::Evaluated => 1,
            Self::Approved => 2,
            Self::Prepared => 3,
            Self::Dispatching => 4,
            Self::Active => 5,
            Self::Partial => 6,
            Self::Purchased => 7,
            Self::Selling => 8,
            Self::Sold | Self::PaperSimulated => 9,
            Self::Rejected
            | Self::Cancelled
            | Self::Superseded
            | Self::Expired
            | Self::Failed
            | Self::Unknown => 100,
        }
    }

    fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Sold
                | Self::PaperSimulated
                | Self::Rejected
                | Self::Cancelled
                | Self::Superseded
                | Self::Expired
                | Self::Failed
                | Self::Unknown
        )
    }
}

fn transition_allowed(from: ReliabilityLifecycleState, to: ReliabilityLifecycleState) -> bool {
    if from == to {
        return true;
    }

    if from.is_terminal() {
        return matches!(
            (from, to),
            (
                ReliabilityLifecycleState::Unknown,
                ReliabilityLifecycleState::Active
            ) | (
                ReliabilityLifecycleState::Unknown,
                ReliabilityLifecycleState::Cancelled
            ) | (
                ReliabilityLifecycleState::Unknown,
                ReliabilityLifecycleState::Failed
            )
        );
    }

    if to.is_terminal() {
        return true;
    }

    to.rank() >= from.rank()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionState {
    Prepared,
    Dispatching,
    Applied,
    Reconciled,
    Blocked,
    Failed,
    Unknown,
    Noop,
}

impl ExecutionState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Prepared => "prepared",
            Self::Dispatching => "dispatching",
            Self::Applied => "applied",
            Self::Reconciled => "reconciled",
            Self::Blocked => "blocked",
            Self::Failed => "failed",
            Self::Unknown => "unknown",
            Self::Noop => "noop",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "prepared" => Self::Prepared,
            "dispatching" => Self::Dispatching,
            "applied" => Self::Applied,
            "reconciled" => Self::Reconciled,
            "blocked" => Self::Blocked,
            "failed" => Self::Failed,
            "noop" => Self::Noop,
            _ => Self::Unknown,
        }
    }

    fn terminal(self) -> bool {
        matches!(
            self,
            Self::Applied | Self::Reconciled | Self::Blocked | Self::Failed | Self::Noop
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ReliabilityConfig {
    pub enabled: bool,
    pub circuit_breaker_enabled: bool,

    pub minimum_model_health: f64,
    pub minimum_data_quality: f64,
    pub maximum_anomaly_score: f64,

    pub execution_failure_window_minutes: i64,
    pub maximum_execution_failures: i64,
    pub maximum_actions_per_minute: i64,
    pub maximum_daily_realized_loss: f64,

    pub prepared_timeout_minutes: i64,
    pub automatic_recovery_minutes: i64,

    pub release_gate_enabled: bool,
    pub release_gate_minimum_samples: usize,
    pub release_gate_minimum_reward_uplift_pct: f64,
    pub release_gate_max_failure_delta: f64,
    pub release_gate_max_drawdown: f64,
    pub release_gate_minimum_walk_forward_stability: f64,
    pub release_gate_require_no_leakage: bool,
    pub release_gate_minimum_safe_stress_fraction: f64,
}

impl Default for ReliabilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            circuit_breaker_enabled: true,

            minimum_model_health: 0.45,
            minimum_data_quality: 0.30,
            maximum_anomaly_score: 0.85,

            execution_failure_window_minutes: 15,
            maximum_execution_failures: 3,
            maximum_actions_per_minute: 12,
            maximum_daily_realized_loss: 250.0,

            prepared_timeout_minutes: 5,
            automatic_recovery_minutes: 30,

            release_gate_enabled: true,
            release_gate_minimum_samples: 50,
            release_gate_minimum_reward_uplift_pct: 0.03,
            release_gate_max_failure_delta: 0.03,
            release_gate_max_drawdown: 0.20,
            release_gate_minimum_walk_forward_stability: 0.55,
            release_gate_require_no_leakage: true,
            release_gate_minimum_safe_stress_fraction: 0.70,
        }
    }
}

impl ReliabilityConfig {
    pub fn normalized(mut self) -> Self {
        self.minimum_model_health = self.minimum_model_health.clamp(0.0, 1.0);
        self.minimum_data_quality = self.minimum_data_quality.clamp(0.0, 1.0);
        self.maximum_anomaly_score = self.maximum_anomaly_score.clamp(0.0, 1.0);
        self.execution_failure_window_minutes =
            self.execution_failure_window_minutes.clamp(1, 24 * 60);
        self.maximum_execution_failures = self.maximum_execution_failures.clamp(1, 100);
        self.maximum_actions_per_minute = self.maximum_actions_per_minute.clamp(1, 120);
        self.maximum_daily_realized_loss = self.maximum_daily_realized_loss.max(0.0);
        self.prepared_timeout_minutes = self.prepared_timeout_minutes.clamp(1, 120);
        self.automatic_recovery_minutes = self.automatic_recovery_minutes.clamp(1, 24 * 60);

        self.release_gate_minimum_samples = self.release_gate_minimum_samples.clamp(10, 100_000);
        self.release_gate_minimum_reward_uplift_pct = self
            .release_gate_minimum_reward_uplift_pct
            .clamp(-0.50, 2.0);
        self.release_gate_max_failure_delta = self.release_gate_max_failure_delta.clamp(0.0, 1.0);
        self.release_gate_max_drawdown = self.release_gate_max_drawdown.clamp(0.0, 1.0);
        self.release_gate_minimum_walk_forward_stability = self
            .release_gate_minimum_walk_forward_stability
            .clamp(0.0, 1.0);
        self.release_gate_minimum_safe_stress_fraction = self
            .release_gate_minimum_safe_stress_fraction
            .clamp(0.0, 1.0);

        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitBreakerStatus {
    pub tripped: bool,
    pub reasons: Vec<String>,
    pub manual_trip: bool,
    pub recent_failures: i64,
    pub recent_actions: i64,
    pub unknown_executions: i64,
    pub daily_realized_loss: f64,
    pub checked_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionPermit {
    pub execution_id: String,
    pub should_execute: bool,
    pub reason: String,
    pub state: ExecutionState,
    pub circuit_breaker: CircuitBreakerStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ExecutionJournalEntry {
    pub id: String,
    pub decision_id: String,
    pub item_key: String,
    pub wfm_id: String,
    pub side: String,
    pub operation: String,
    pub target_price: i64,
    pub target_quantity: i64,
    pub previous_price: i64,
    pub state: String,
    pub attempt_count: i64,
    pub error: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReliabilityStatus {
    pub config: ReliabilityConfig,
    pub breaker: CircuitBreakerStatus,
    pub recent_executions: Vec<ExecutionJournalEntry>,
    pub pending_executions: i64,
    pub lifecycle_events: i64,
    pub last_release_gate: Option<ReleaseGateReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct FakeMarketScenarioResult {
    pub name: String,
    pub passed: bool,
    pub final_state: String,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FakeMarketSuiteReport {
    pub passed: bool,
    pub passed_count: usize,
    pub total_count: usize,
    pub scenarios: Vec<FakeMarketScenarioResult>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseGateReport {
    pub id: String,
    pub champion_policy: String,
    pub candidate_policy: String,
    pub passed: bool,
    pub reasons: Vec<String>,

    pub champion: OfflinePolicyEvaluation,
    pub candidate: OfflinePolicyEvaluation,

    pub reward_uplift_pct: f64,
    pub failure_delta: f64,
    pub drawdown_delta: f64,

    pub walk_forward_stability: f64,
    pub leakage_healthy: bool,
    pub safe_stress_fraction: f64,

    pub created_at: DateTime<Utc>,
}

pub struct ReliabilityService;

impl ReliabilityService {
    pub async fn load_config(db: &DatabaseConnection) -> Result<ReliabilityConfig, Error> {
        let row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT value
                 FROM algoframe_setting
                 WHERE key='reliability_config'
                 LIMIT 1"
                    .to_string(),
            ))
            .await
            .map_err(|error| rel_error("LoadConfig", error))?;

        if let Some(row) = row {
            let value: String = row.try_get("", "value").unwrap_or_default();
            if let Ok(config) = serde_json::from_str::<ReliabilityConfig>(&value) {
                return Ok(config.normalized());
            }
        }

        let config = ReliabilityConfig::default();
        Self::save_config(db, &config).await?;
        Ok(config)
    }

    pub async fn save_config(
        db: &DatabaseConnection,
        config: &ReliabilityConfig,
    ) -> Result<(), Error> {
        let config = config.clone().normalized();
        let value = serde_json::to_string(&config)
            .map_err(|error| rel_error("SaveConfig:Serialize", error))?;

        execute(
            db,
            "SaveConfig",
            format!(
                "INSERT INTO algoframe_setting(key, value, updated_at)
                 VALUES('reliability_config', '{}', '{}')
                 ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at",
                sql_string(&value),
                Utc::now().to_rfc3339(),
            ),
        )
        .await
    }

    async fn manual_trip_reason(db: &DatabaseConnection) -> Result<Option<String>, Error> {
        let row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT value
                 FROM algoframe_setting
                 WHERE key='reliability_manual_trip'
                 LIMIT 1"
                    .to_string(),
            ))
            .await
            .map_err(|error| rel_error("ManualTrip", error))?;

        let Some(row) = row else {
            return Ok(None);
        };

        let value: String = row.try_get("", "value").unwrap_or_default();
        if value.trim().is_empty() {
            return Ok(None);
        }

        Ok(Some(value))
    }

    pub async fn trip_manual(db: &DatabaseConnection, reason: &str) -> Result<(), Error> {
        execute(
            db,
            "TripManual",
            format!(
                "INSERT INTO algoframe_setting(key, value, updated_at)
                 VALUES('reliability_manual_trip', '{}', '{}')
                 ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at",
                sql_string(reason),
                Utc::now().to_rfc3339(),
            ),
        )
        .await?;

        Self::record_circuit_event(db, "manual_trip", true, reason).await
    }

    pub async fn clear_manual(db: &DatabaseConnection) -> Result<(), Error> {
        execute(
            db,
            "ClearManual",
            "DELETE FROM algoframe_setting
             WHERE key='reliability_manual_trip'"
                .to_string(),
        )
        .await?;

        Self::record_circuit_event(db, "manual_clear", false, "manual circuit breaker cleared")
            .await
    }

    async fn record_circuit_event(
        db: &DatabaseConnection,
        code: &str,
        tripped: bool,
        reason: &str,
    ) -> Result<(), Error> {
        let id = format!(
            "circuit:{}:{}",
            Utc::now().timestamp_millis(),
            stable_text_hash(&format!("{code}|{reason}"))
        );

        execute(
            db,
            "CircuitEvent",
            format!(
                "INSERT INTO algoframe_circuit_event(
                    id, code, tripped, reason, created_at
                 ) VALUES('{}','{}',{},{},'{}')",
                sql_string(&id),
                sql_string(code),
                if tripped { 1 } else { 0 },
                format!("'{}'", sql_string(reason)),
                Utc::now().to_rfc3339(),
            ),
        )
        .await
    }

    pub async fn circuit_breaker_status(
        db: &DatabaseConnection,
        snapshot: Option<&MarketSnapshot>,
        health: Option<&ModelHealth>,
    ) -> Result<CircuitBreakerStatus, Error> {
        let config = Self::load_config(db).await?;
        let manual_reason = Self::manual_trip_reason(db).await?;

        let recent_failure_cutoff =
            Utc::now() - Duration::minutes(config.execution_failure_window_minutes);

        let failure_row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                format!(
                    "SELECT COUNT(*) AS count
                     FROM algoframe_execution_journal
                     WHERE state IN ('failed','unknown')
                       AND updated_at >= '{}'",
                    recent_failure_cutoff.to_rfc3339()
                ),
            ))
            .await
            .map_err(|error| rel_error("Breaker:Failures", error))?;

        let recent_failures = failure_row
            .and_then(|row| row.try_get("", "count").ok())
            .unwrap_or(0_i64);

        let action_cutoff = Utc::now() - Duration::minutes(1);

        let action_row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                format!(
                    "SELECT COUNT(*) AS count
                     FROM algoframe_execution_journal
                     WHERE state IN ('dispatching','applied','reconciled')
                       AND updated_at >= '{}'",
                    action_cutoff.to_rfc3339()
                ),
            ))
            .await
            .map_err(|error| rel_error("Breaker:Actions", error))?;

        let recent_actions = action_row
            .and_then(|row| row.try_get("", "count").ok())
            .unwrap_or(0_i64);

        let unknown_row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT COUNT(*) AS count
                 FROM algoframe_execution_journal
                 WHERE state='unknown'"
                    .to_string(),
            ))
            .await
            .map_err(|error| rel_error("Breaker:Unknown", error))?;

        let unknown_executions = unknown_row
            .and_then(|row| row.try_get("", "count").ok())
            .unwrap_or(0_i64);

        let loss_row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT COALESCE(
                    SUM(
                        CASE
                            WHEN actual_profit < 0
                            THEN ABS(actual_profit * quantity)
                            ELSE 0
                        END
                    ),
                    0
                 ) AS loss
                 FROM algoframe_decision
                 WHERE date(updated_at) = date('now')
                   AND status IN ('completed','paper_completed')"
                    .to_string(),
            ))
            .await
            .map_err(|error| rel_error("Breaker:DailyLoss", error))?;

        let daily_realized_loss = loss_row
            .and_then(|row| row.try_get("", "loss").ok())
            .unwrap_or(0.0_f64);

        let mut reasons = Vec::new();

        if let Some(reason) = &manual_reason {
            reasons.push(format!("manual freeze: {reason}"));
        }

        if let Some(health) = health {
            if health.score < config.minimum_model_health {
                reasons.push(format!(
                    "model health {:.0}% is below reliability floor {:.0}%",
                    health.score * 100.0,
                    config.minimum_model_health * 100.0,
                ));
            }

            if health.fallback_active {
                reasons.push("model fallback is active".to_string());
            }
        }

        if let Some(snapshot) = snapshot {
            if snapshot.features.quality.score < config.minimum_data_quality {
                reasons.push(format!(
                    "market data quality {:.0}% is below {:.0}%",
                    snapshot.features.quality.score * 100.0,
                    config.minimum_data_quality * 100.0,
                ));
            }

            if snapshot.features.anomaly.score > config.maximum_anomaly_score {
                reasons.push(format!(
                    "market anomaly score {:.0}% exceeds {:.0}%",
                    snapshot.features.anomaly.score * 100.0,
                    config.maximum_anomaly_score * 100.0,
                ));
            }
        }

        if recent_failures >= config.maximum_execution_failures {
            reasons.push(format!(
                "{recent_failures} execution failures/unknowns in {} minutes",
                config.execution_failure_window_minutes
            ));
        }

        if recent_actions >= config.maximum_actions_per_minute {
            reasons.push(format!(
                "{recent_actions} execution actions in the last minute"
            ));
        }

        if unknown_executions > 0 {
            reasons.push(format!(
                "{unknown_executions} execution(s) have uncertain remote state"
            ));
        }

        if config.maximum_daily_realized_loss > 0.0
            && daily_realized_loss >= config.maximum_daily_realized_loss
        {
            reasons.push(format!(
                "daily realized loss {:.1}p reached safety limit {:.1}p",
                daily_realized_loss, config.maximum_daily_realized_loss
            ));
        }

        let tripped = config.enabled && config.circuit_breaker_enabled && !reasons.is_empty();

        Ok(CircuitBreakerStatus {
            tripped,
            reasons,
            manual_trip: manual_reason.is_some(),
            recent_failures,
            recent_actions,
            unknown_executions,
            daily_realized_loss,
            checked_at: Utc::now(),
        })
    }

    pub async fn record_intent(
        db: &DatabaseConnection,
        intent: &ExecutionIntent,
    ) -> Result<(), Error> {
        let side = side_string(intent.side);

        Self::transition(
            db,
            &intent.decision_id,
            &intent.item_key,
            side,
            ReliabilityLifecycleState::Discovered,
            "AlgoFrame observed a tradeable market state",
            serde_json::json!({
                "snapshot_id": intent.snapshot_id,
                "wfm_id": intent.wfm_id,
            }),
        )
        .await?;

        Self::transition(
            db,
            &intent.decision_id,
            &intent.item_key,
            side,
            ReliabilityLifecycleState::Evaluated,
            "AlgoFrame evaluated the market opportunity",
            serde_json::json!({
                "selected_action": intent.selected_action,
                "price": intent.price,
                "quantity": intent.quantity,
                "confidence": intent.confidence,
            }),
        )
        .await?;

        let target = if intent.paper {
            ReliabilityLifecycleState::PaperSimulated
        } else if intent.allowed {
            ReliabilityLifecycleState::Approved
        } else {
            ReliabilityLifecycleState::Rejected
        };

        Self::transition(
            db,
            &intent.decision_id,
            &intent.item_key,
            side,
            target,
            if intent.allowed {
                "decision approved by policy and guardrails"
            } else {
                "decision rejected by policy or guardrails"
            },
            serde_json::json!({
                "allowed": intent.allowed,
                "paper": intent.paper,
                "guardrails": intent.explanation.guardrails,
                "negative_factors": intent.explanation.negative_factors,
            }),
        )
        .await
    }

    async fn latest_lifecycle(
        db: &DatabaseConnection,
        decision_id: &str,
    ) -> Result<Option<ReliabilityLifecycleState>, Error> {
        let row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                format!(
                    "SELECT to_state
                     FROM algoframe_lifecycle_event
                     WHERE decision_id='{}'
                     ORDER BY created_at DESC
                     LIMIT 1",
                    sql_string(decision_id)
                ),
            ))
            .await
            .map_err(|error| rel_error("Lifecycle:Latest", error))?;

        Ok(row
            .and_then(|row| row.try_get::<String>("", "to_state").ok())
            .map(|state| ReliabilityLifecycleState::parse(&state)))
    }

    pub async fn transition(
        db: &DatabaseConnection,
        decision_id: &str,
        item_key: &str,
        side: &str,
        to: ReliabilityLifecycleState,
        reason: &str,
        payload: serde_json::Value,
    ) -> Result<(), Error> {
        let previous = Self::latest_lifecycle(db, decision_id).await?;

        if previous == Some(to) {
            return Ok(());
        }

        if let Some(from) = previous {
            if !transition_allowed(from, to) {
                return Err(rel_error(
                    "Lifecycle:InvalidTransition",
                    format!(
                        "invalid transition {} -> {} for decision {}",
                        from.as_str(),
                        to.as_str(),
                        decision_id
                    ),
                ));
            }
        }

        let now = Utc::now();
        let id = format!(
            "life:{}:{}:{}",
            decision_id,
            now.timestamp_millis(),
            stable_text_hash(&format!("{reason}|{}", to.as_str()))
        );
        let payload = serde_json::to_string(&payload)
            .map_err(|error| rel_error("Lifecycle:Serialize", error))?;

        execute(
            db,
            "Lifecycle:Insert",
            format!(
                "INSERT INTO algoframe_lifecycle_event(
                    id, decision_id, item_key, side,
                    from_state, to_state, reason, created_at, payload
                 ) VALUES(
                    '{}','{}','{}','{}','{}','{}','{}','{}','{}'
                 )",
                sql_string(&id),
                sql_string(decision_id),
                sql_string(item_key),
                sql_string(side),
                previous.map(|state| state.as_str()).unwrap_or(""),
                to.as_str(),
                sql_string(reason),
                now.to_rfc3339(),
                sql_string(&payload),
            ),
        )
        .await
    }

    pub async fn prepare_execution(
        db: &DatabaseConnection,
        intent: &ExecutionIntent,
        operation: &str,
        target_price: i64,
        target_quantity: i64,
        previous_price: i64,
        snapshot: &MarketSnapshot,
        health: &ModelHealth,
    ) -> Result<ExecutionPermit, Error> {
        let breaker = Self::circuit_breaker_status(db, Some(snapshot), Some(health)).await?;

        let id = execution_id(
            &intent.decision_id,
            intent.side,
            operation,
            target_price,
            target_quantity,
        );

        if breaker.tripped {
            Self::upsert_execution(
                db,
                &id,
                intent,
                operation,
                target_price,
                target_quantity,
                previous_price,
                ExecutionState::Blocked,
                0,
                &breaker.reasons.join("; "),
            )
            .await?;

            let _ = Self::transition(
                db,
                &intent.decision_id,
                &intent.item_key,
                side_string(intent.side),
                ReliabilityLifecycleState::Cancelled,
                "execution prevented by circuit breaker",
                serde_json::json!({ "reasons": breaker.reasons }),
            )
            .await;

            return Ok(ExecutionPermit {
                execution_id: id,
                should_execute: false,
                reason: "circuit breaker is tripped".to_string(),
                state: ExecutionState::Blocked,
                circuit_breaker: breaker,
            });
        }

        let mut next_attempt = 1_i64;

        if let Some(existing) = Self::load_execution(db, &id).await? {
            let state = ExecutionState::parse(&existing.state);
            next_attempt = existing.attempt_count.saturating_add(1).max(1);

            if matches!(state, ExecutionState::Applied | ExecutionState::Reconciled) {
                return Ok(ExecutionPermit {
                    execution_id: id,
                    should_execute: false,
                    reason: "identical execution was already applied".to_string(),
                    state,
                    circuit_breaker: breaker,
                });
            }

            if state == ExecutionState::Unknown {
                return Ok(ExecutionPermit {
                    execution_id: id,
                    should_execute: false,
                    reason:
                        "identical execution has unknown remote state and cannot be retried automatically"
                            .to_string(),
                    state,
                    circuit_breaker: breaker,
                });
            }

            if matches!(
                state,
                ExecutionState::Prepared | ExecutionState::Dispatching
            ) {
                let age = DateTime::parse_from_rfc3339(&existing.updated_at)
                    .ok()
                    .map(|time| Utc::now() - time.with_timezone(&Utc))
                    .unwrap_or_else(Duration::zero);

                let config = Self::load_config(db).await?;

                if age < Duration::minutes(config.prepared_timeout_minutes) {
                    return Ok(ExecutionPermit {
                        execution_id: id,
                        should_execute: false,
                        reason: "identical execution is already in flight".to_string(),
                        state,
                        circuit_breaker: breaker,
                    });
                }

                Self::set_execution_state(
                    db,
                    &id,
                    ExecutionState::Unknown,
                    "stale prepared/dispatching execution requires reconciliation",
                )
                .await?;

                return Ok(ExecutionPermit {
                    execution_id: id,
                    should_execute: false,
                    reason:
                        "stale execution has uncertain remote state; automation frozen until reconciliation"
                            .to_string(),
                    state: ExecutionState::Unknown,
                    circuit_breaker: Self::circuit_breaker_status(
                        db,
                        Some(snapshot),
                        Some(health),
                    )
                    .await?,
                });
            }
        }

        Self::upsert_execution(
            db,
            &id,
            intent,
            operation,
            target_price,
            target_quantity,
            previous_price,
            ExecutionState::Prepared,
            next_attempt,
            "",
        )
        .await?;

        if intent.allowed {
            Self::transition(
                db,
                &intent.decision_id,
                &intent.item_key,
                side_string(intent.side),
                ReliabilityLifecycleState::Prepared,
                "execution intent persisted before remote mutation",
                serde_json::json!({
                    "execution_id": id,
                    "operation": operation,
                    "target_price": target_price,
                    "target_quantity": target_quantity,
                }),
            )
            .await?;
        }

        Ok(ExecutionPermit {
            execution_id: id,
            should_execute: true,
            reason: "execution prepared".to_string(),
            state: ExecutionState::Prepared,
            circuit_breaker: breaker,
        })
    }

    async fn upsert_execution(
        db: &DatabaseConnection,
        id: &str,
        intent: &ExecutionIntent,
        operation: &str,
        target_price: i64,
        target_quantity: i64,
        previous_price: i64,
        state: ExecutionState,
        attempt_count: i64,
        error: &str,
    ) -> Result<(), Error> {
        let now = Utc::now();
        let payload = serde_json::json!({
            "snapshot_id": intent.snapshot_id,
            "item_name": intent.item_name,
            "category": intent.category,
            "policy_name": intent.policy_name,
            "model_version": intent.model_version,
        });

        execute(
            db,
            "Execution:Upsert",
            format!(
                "INSERT INTO algoframe_execution_journal(
                    id, decision_id, item_key, wfm_id, side, operation,
                    target_price, target_quantity, previous_price,
                    state, attempt_count, error, created_at, updated_at, payload
                 ) VALUES(
                    '{}','{}','{}','{}','{}','{}',
                    {},{},{},'{}',{},'{}','{}','{}','{}'
                 )
                 ON CONFLICT(id) DO UPDATE SET
                    state=excluded.state,
                    attempt_count=MAX(algoframe_execution_journal.attempt_count, excluded.attempt_count),
                    error=excluded.error,
                    updated_at=excluded.updated_at,
                    payload=excluded.payload",
                sql_string(id),
                sql_string(&intent.decision_id),
                sql_string(&intent.item_key),
                sql_string(&intent.wfm_id),
                side_string(intent.side),
                sql_string(operation),
                target_price,
                target_quantity,
                previous_price,
                state.as_str(),
                attempt_count,
                sql_string(error),
                now.to_rfc3339(),
                now.to_rfc3339(),
                sql_string(&payload.to_string()),
            ),
        )
        .await
    }

    pub async fn mark_dispatching(
        db: &DatabaseConnection,
        execution_id: &str,
    ) -> Result<(), Error> {
        Self::set_execution_state(db, execution_id, ExecutionState::Dispatching, "").await
    }

    pub async fn mark_applied(db: &DatabaseConnection, execution_id: &str) -> Result<(), Error> {
        let entry = Self::load_execution(db, execution_id)
            .await?
            .ok_or_else(|| rel_error("Execution:Applied", "execution was not found"))?;

        Self::set_execution_state(db, execution_id, ExecutionState::Applied, "").await?;

        let target_state = if entry.operation == "delete" {
            ReliabilityLifecycleState::Cancelled
        } else {
            ReliabilityLifecycleState::Active
        };

        // Deleting an order that belongs to a policy-rejected decision may
        // already have a terminal Rejected lifecycle. The execution journal
        // remains authoritative even if that lifecycle transition is a no-op.
        let _ = Self::transition(
            db,
            &entry.decision_id,
            &entry.item_key,
            &entry.side,
            target_state,
            "remote order mutation returned success",
            serde_json::json!({
                "execution_id": execution_id,
                "operation": entry.operation,
                "price": entry.target_price,
                "quantity": entry.target_quantity,
            }),
        )
        .await;

        Ok(())
    }

    pub async fn mark_noop(
        db: &DatabaseConnection,
        execution_id: &str,
        reason: &str,
    ) -> Result<(), Error> {
        Self::set_execution_state(db, execution_id, ExecutionState::Noop, reason).await
    }

    pub async fn mark_uncertain(
        db: &DatabaseConnection,
        execution_id: &str,
        error: &str,
    ) -> Result<(), Error> {
        let entry = Self::load_execution(db, execution_id).await?;

        Self::set_execution_state(db, execution_id, ExecutionState::Unknown, error).await?;

        if let Some(entry) = entry {
            let _ = Self::transition(
                db,
                &entry.decision_id,
                &entry.item_key,
                &entry.side,
                ReliabilityLifecycleState::Unknown,
                "remote mutation returned an error with uncertain remote state",
                serde_json::json!({
                    "execution_id": execution_id,
                    "error": error,
                }),
            )
            .await;
        }

        Self::record_circuit_event(
            db,
            "execution_unknown",
            true,
            &format!("execution {execution_id}: {error}"),
        )
        .await
    }

    async fn set_execution_state(
        db: &DatabaseConnection,
        id: &str,
        state: ExecutionState,
        error: &str,
    ) -> Result<(), Error> {
        execute(
            db,
            "Execution:SetState",
            format!(
                "UPDATE algoframe_execution_journal
                 SET state='{}',
                     error='{}',
                     updated_at='{}'
                 WHERE id='{}'",
                state.as_str(),
                sql_string(error),
                Utc::now().to_rfc3339(),
                sql_string(id),
            ),
        )
        .await
    }

    async fn load_execution(
        db: &DatabaseConnection,
        id: &str,
    ) -> Result<Option<ExecutionJournalEntry>, Error> {
        let row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                format!(
                    "SELECT
                        id, decision_id, item_key, wfm_id, side, operation,
                        target_price, target_quantity, previous_price,
                        state, attempt_count, error, created_at, updated_at
                     FROM algoframe_execution_journal
                     WHERE id='{}'
                     LIMIT 1",
                    sql_string(id)
                ),
            ))
            .await
            .map_err(|error| rel_error("Execution:Load", error))?;

        Ok(row.map(row_to_execution))
    }

    pub async fn list_executions(
        db: &DatabaseConnection,
        limit: usize,
    ) -> Result<Vec<ExecutionJournalEntry>, Error> {
        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Sqlite,
                format!(
                    "SELECT
                        id, decision_id, item_key, wfm_id, side, operation,
                        target_price, target_quantity, previous_price,
                        state, attempt_count, error, created_at, updated_at
                     FROM algoframe_execution_journal
                     ORDER BY updated_at DESC
                     LIMIT {}",
                    limit.clamp(1, 5000)
                ),
            ))
            .await
            .map_err(|error| rel_error("Execution:List", error))?;

        Ok(rows.into_iter().map(row_to_execution).collect())
    }

    pub async fn reconcile_startup(
        db: &DatabaseConnection,
        my_orders: &OrderList<Order>,
    ) -> Result<(), Error> {
        let config = Self::load_config(db).await?;

        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT
                    id, decision_id, item_key, wfm_id, side, operation,
                    target_price, target_quantity, previous_price,
                    state, attempt_count, error, created_at, updated_at
                 FROM algoframe_execution_journal
                 WHERE state IN ('prepared','dispatching','unknown')
                 ORDER BY updated_at ASC"
                    .to_string(),
            ))
            .await
            .map_err(|error| rel_error("Reconcile:Load", error))?;

        for row in rows {
            let entry = row_to_execution(row);
            let updated = DateTime::parse_from_rfc3339(&entry.updated_at)
                .ok()
                .map(|time| time.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            if Utc::now() - updated < Duration::minutes(config.prepared_timeout_minutes)
                && entry.state != "unknown"
            {
                continue;
            }

            let orders = if entry.side == "sell" {
                &my_orders.sell_orders
            } else {
                &my_orders.buy_orders
            };

            let matching_decision = orders.iter().find(|order| {
                order
                    .properties
                    .get_property_value("algoframe_decision_id", String::new())
                    == entry.decision_id
            });

            let same_item = orders.iter().find(|order| order.item_id == entry.wfm_id);

            let reconciled = if entry.operation == "delete" {
                same_item.is_none()
            } else {
                matching_decision
                    .map(|order| {
                        order.platinum as i64 == entry.target_price
                            && order.quantity as i64 == entry.target_quantity
                    })
                    .unwrap_or(false)
            };

            if reconciled {
                Self::set_execution_state(
                    db,
                    &entry.id,
                    ExecutionState::Reconciled,
                    "remote order state matched prepared execution after restart",
                )
                .await?;

                Self::transition(
                    db,
                    &entry.decision_id,
                    &entry.item_key,
                    &entry.side,
                    ReliabilityLifecycleState::Active,
                    "execution reconciled against current WFM order cache",
                    serde_json::json!({
                        "execution_id": entry.id,
                        "reconciled": true,
                    }),
                )
                .await?;
            } else {
                Self::set_execution_state(
                    db,
                    &entry.id,
                    ExecutionState::Unknown,
                    "remote state did not prove whether the prepared mutation was applied",
                )
                .await?;

                Self::record_circuit_event(
                    db,
                    "reconciliation_unknown",
                    true,
                    &format!(
                        "execution {} for decision {} could not be reconciled",
                        entry.id, entry.decision_id
                    ),
                )
                .await?;
            }
        }

        Ok(())
    }

    pub async fn sync_from_decisions(
        db: &DatabaseConnection,
        decisions: &[DecisionRecord],
    ) -> Result<(), Error> {
        for decision in decisions.iter().rev().take(5000) {
            let state = match decision.status {
                DecisionStatus::Open => {
                    if decision.side == TradeSide::Sell {
                        Some(ReliabilityLifecycleState::Selling)
                    } else {
                        Some(ReliabilityLifecycleState::Active)
                    }
                }
                DecisionStatus::Partial => Some(ReliabilityLifecycleState::Partial),
                DecisionStatus::Filled => {
                    if decision.side == TradeSide::Buy {
                        Some(ReliabilityLifecycleState::Purchased)
                    } else {
                        Some(ReliabilityLifecycleState::Active)
                    }
                }
                DecisionStatus::Completed => Some(ReliabilityLifecycleState::Sold),
                DecisionStatus::PaperOpen | DecisionStatus::PaperCompleted => {
                    Some(ReliabilityLifecycleState::PaperSimulated)
                }
                DecisionStatus::Expired => Some(ReliabilityLifecycleState::Expired),
                DecisionStatus::Rejected => Some(ReliabilityLifecycleState::Rejected),
                DecisionStatus::Cancelled => Some(ReliabilityLifecycleState::Cancelled),
            };

            if let Some(state) = state {
                let _ = Self::transition(
                    db,
                    &decision.id,
                    &decision.item_key,
                    side_string(decision.side),
                    state,
                    "synchronized from persisted AlgoFrame decision state",
                    serde_json::json!({
                        "decision_status": decision.status,
                        "filled_quantity": decision.filled_quantity,
                        "fill_ratio": decision.fill_ratio,
                    }),
                )
                .await;
            }
        }

        Ok(())
    }

    pub async fn resolve_execution(
        db: &DatabaseConnection,
        execution_id: &str,
        resolution: &str,
    ) -> Result<(), Error> {
        let entry = Self::load_execution(db, execution_id)
            .await?
            .ok_or_else(|| rel_error("Execution:Resolve", "execution was not found"))?;

        if ExecutionState::parse(&entry.state) != ExecutionState::Unknown {
            return Err(rel_error(
                "Execution:Resolve",
                "only Unknown executions require manual resolution",
            ));
        }

        match resolution {
            "applied" => {
                Self::set_execution_state(
                    db,
                    execution_id,
                    ExecutionState::Reconciled,
                    "operator verified that the remote mutation was applied",
                )
                .await?;

                let _ = Self::transition(
                    db,
                    &entry.decision_id,
                    &entry.item_key,
                    &entry.side,
                    ReliabilityLifecycleState::Active,
                    "operator verified unknown execution as applied",
                    serde_json::json!({ "execution_id": execution_id }),
                )
                .await;
            }
            "retry" => {
                Self::set_execution_state(
                    db,
                    execution_id,
                    ExecutionState::Noop,
                    "operator verified remote state and cleared this execution for a future retry",
                )
                .await?;
            }
            "cancelled" => {
                Self::set_execution_state(
                    db,
                    execution_id,
                    ExecutionState::Failed,
                    "operator resolved unknown execution as not applied/cancelled",
                )
                .await?;

                let _ = Self::transition(
                    db,
                    &entry.decision_id,
                    &entry.item_key,
                    &entry.side,
                    ReliabilityLifecycleState::Cancelled,
                    "operator resolved unknown execution as cancelled",
                    serde_json::json!({ "execution_id": execution_id }),
                )
                .await;
            }
            _ => {
                return Err(rel_error(
                    "Execution:Resolve",
                    "resolution must be 'applied', 'retry', or 'cancelled'",
                ));
            }
        }

        Self::record_circuit_event(
            db,
            "execution_resolved",
            false,
            &format!(
                "execution {} manually resolved as {}",
                execution_id, resolution
            ),
        )
        .await
    }

    pub async fn get_status(db: &DatabaseConnection) -> Result<ReliabilityStatus, Error> {
        let config = Self::load_config(db).await?;
        let breaker = Self::circuit_breaker_status(db, None, None).await?;
        let recent_executions = Self::list_executions(db, 100).await?;

        let pending = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT COUNT(*) AS count
                 FROM algoframe_execution_journal
                 WHERE state IN ('prepared','dispatching','unknown')"
                    .to_string(),
            ))
            .await
            .map_err(|error| rel_error("Status:Pending", error))?
            .and_then(|row| row.try_get("", "count").ok())
            .unwrap_or(0_i64);

        let lifecycle_events = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT COUNT(*) AS count FROM algoframe_lifecycle_event".to_string(),
            ))
            .await
            .map_err(|error| rel_error("Status:Lifecycle", error))?
            .and_then(|row| row.try_get("", "count").ok())
            .unwrap_or(0_i64);

        let last_release_gate = Self::load_last_release_gate(db).await?;

        Ok(ReliabilityStatus {
            config,
            breaker,
            recent_executions,
            pending_executions: pending,
            lifecycle_events,
            last_release_gate,
        })
    }

    pub fn evaluate_release_gate(
        decisions: &[DecisionRecord],
        snapshots: &[MarketSnapshot],
        champion_policy: &str,
        candidate_policy: &str,
        config: &ReliabilityConfig,
    ) -> ReleaseGateReport {
        let champion_name = if champion_policy.starts_with("shadow_") {
            champion_policy.to_string()
        } else {
            format!("shadow_{champion_policy}")
        };

        let candidate_name = if candidate_policy.starts_with("shadow_") {
            candidate_policy.to_string()
        } else {
            format!("shadow_{candidate_policy}")
        };

        let champion = offline_policy_evaluation(decisions, &champion_name);
        let candidate = offline_policy_evaluation(decisions, &candidate_name);

        let champion_reward = champion
            .metrics
            .doubly_robust_reward
            .max(champion.metrics.average_reward);
        let candidate_reward = candidate
            .metrics
            .doubly_robust_reward
            .max(candidate.metrics.average_reward);

        let reward_uplift_pct = if champion_reward.abs() > 1e-9 {
            (candidate_reward - champion_reward) / champion_reward.abs()
        } else {
            candidate_reward - champion_reward
        };

        let failure_delta = candidate.metrics.failure_rate - champion.metrics.failure_rate;
        let drawdown_delta = candidate.metrics.max_drawdown - champion.metrics.max_drawdown;

        let walk_forward = walk_forward_validation(decisions);
        let leakage = leakage_report(decisions, snapshots);
        let stress = synthetic_stress_tests(decisions);

        let safe_count = stress.iter().filter(|result| result.safe).count();
        let safe_stress_fraction = if stress.is_empty() {
            0.0
        } else {
            safe_count as f64 / stress.len() as f64
        };

        let mut reasons = Vec::new();

        if candidate.metrics.sample_count < config.release_gate_minimum_samples {
            reasons.push(format!(
                "candidate has {} samples; {} required",
                candidate.metrics.sample_count, config.release_gate_minimum_samples
            ));
        }

        if reward_uplift_pct < config.release_gate_minimum_reward_uplift_pct {
            reasons.push(format!(
                "reward uplift {:.1}% is below required {:.1}%",
                reward_uplift_pct * 100.0,
                config.release_gate_minimum_reward_uplift_pct * 100.0
            ));
        }

        if failure_delta > config.release_gate_max_failure_delta {
            reasons.push(format!(
                "failure-rate delta {:.1}% exceeds {:.1}%",
                failure_delta * 100.0,
                config.release_gate_max_failure_delta * 100.0
            ));
        }

        if candidate.metrics.max_drawdown > config.release_gate_max_drawdown {
            reasons.push(format!(
                "candidate drawdown {:.1}% exceeds {:.1}%",
                candidate.metrics.max_drawdown * 100.0,
                config.release_gate_max_drawdown * 100.0
            ));
        }

        if candidate.confidence_low <= champion.confidence_high {
            reasons
                .push("candidate confidence interval does not cleanly beat champion".to_string());
        }

        if walk_forward.stability < config.release_gate_minimum_walk_forward_stability {
            reasons.push(format!(
                "walk-forward stability {:.0}% is below {:.0}%",
                walk_forward.stability * 100.0,
                config.release_gate_minimum_walk_forward_stability * 100.0
            ));
        }

        if config.release_gate_require_no_leakage && !leakage.healthy {
            reasons.push("leakage detector reported suspicious observations".to_string());
        }

        if safe_stress_fraction < config.release_gate_minimum_safe_stress_fraction {
            reasons.push(format!(
                "only {:.0}% of stress scenarios passed; {:.0}% required",
                safe_stress_fraction * 100.0,
                config.release_gate_minimum_safe_stress_fraction * 100.0
            ));
        }

        let passed = !config.release_gate_enabled || reasons.is_empty();
        let now = Utc::now();

        ReleaseGateReport {
            id: format!(
                "release_gate:{}:{}",
                now.timestamp_millis(),
                stable_text_hash(&format!("{champion_policy}|{candidate_policy}"))
            ),
            champion_policy: champion_policy.to_string(),
            candidate_policy: candidate_policy.to_string(),
            passed,
            reasons,
            champion,
            candidate,
            reward_uplift_pct,
            failure_delta,
            drawdown_delta,
            walk_forward_stability: walk_forward.stability,
            leakage_healthy: leakage.healthy,
            safe_stress_fraction,
            created_at: now,
        }
    }

    pub async fn store_release_gate(
        db: &DatabaseConnection,
        report: &ReleaseGateReport,
    ) -> Result<(), Error> {
        let payload = serde_json::to_string(report)
            .map_err(|error| rel_error("ReleaseGate:Serialize", error))?;

        execute(
            db,
            "ReleaseGate:Insert",
            format!(
                "INSERT OR REPLACE INTO algoframe_release_gate(
                    id, champion_policy, candidate_policy, passed, created_at, payload
                 ) VALUES('{}','{}','{}',{},'{}','{}')",
                sql_string(&report.id),
                sql_string(&report.champion_policy),
                sql_string(&report.candidate_policy),
                if report.passed { 1 } else { 0 },
                report.created_at.to_rfc3339(),
                sql_string(&payload),
            ),
        )
        .await
    }

    pub async fn load_last_release_gate(
        db: &DatabaseConnection,
    ) -> Result<Option<ReleaseGateReport>, Error> {
        let row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT payload
                 FROM algoframe_release_gate
                 ORDER BY created_at DESC
                 LIMIT 1"
                    .to_string(),
            ))
            .await
            .map_err(|error| rel_error("ReleaseGate:Load", error))?;

        let Some(row) = row else {
            return Ok(None);
        };

        let payload: String = row.try_get("", "payload").unwrap_or_default();
        Ok(serde_json::from_str(&payload).ok())
    }

    pub async fn run_release_gate(
        db: &DatabaseConnection,
        candidate_policy: Option<&str>,
    ) -> Result<ReleaseGateReport, Error> {
        let decisions = LearningStore::load_decisions(db, 50_000).await?;
        let snapshots = LearningStore::load_recent_snapshots(db, 50_000).await?;
        let ultimate = LearningStore::load_config(db).await?;
        let reliability = Self::load_config(db).await?;

        let candidate = candidate_policy.unwrap_or(&ultimate.challenger_policy);

        let report = Self::evaluate_release_gate(
            &decisions,
            &snapshots,
            &ultimate.champion_policy,
            candidate,
            &reliability,
        );

        Self::store_release_gate(db, &report).await?;
        Ok(report)
    }

    pub fn run_fake_market_suite() -> FakeMarketSuiteReport {
        let mut scenarios = Vec::new();

        scenarios.push(run_fake_scenario(
            "happy path buy -> fill -> sell",
            &[
                ReliabilityLifecycleState::Discovered,
                ReliabilityLifecycleState::Evaluated,
                ReliabilityLifecycleState::Approved,
                ReliabilityLifecycleState::Prepared,
                ReliabilityLifecycleState::Dispatching,
                ReliabilityLifecycleState::Active,
                ReliabilityLifecycleState::Purchased,
                ReliabilityLifecycleState::Selling,
                ReliabilityLifecycleState::Sold,
            ],
            ReliabilityLifecycleState::Sold,
        ));

        scenarios.push(run_fake_scenario(
            "partial fill before completion",
            &[
                ReliabilityLifecycleState::Discovered,
                ReliabilityLifecycleState::Evaluated,
                ReliabilityLifecycleState::Approved,
                ReliabilityLifecycleState::Prepared,
                ReliabilityLifecycleState::Dispatching,
                ReliabilityLifecycleState::Active,
                ReliabilityLifecycleState::Partial,
                ReliabilityLifecycleState::Purchased,
                ReliabilityLifecycleState::Selling,
                ReliabilityLifecycleState::Sold,
            ],
            ReliabilityLifecycleState::Sold,
        ));

        scenarios.push(run_fake_scenario(
            "rejected opportunity",
            &[
                ReliabilityLifecycleState::Discovered,
                ReliabilityLifecycleState::Evaluated,
                ReliabilityLifecycleState::Rejected,
            ],
            ReliabilityLifecycleState::Rejected,
        ));

        scenarios.push(run_fake_scenario(
            "crash after dispatch becomes unknown instead of duplicate retry",
            &[
                ReliabilityLifecycleState::Discovered,
                ReliabilityLifecycleState::Evaluated,
                ReliabilityLifecycleState::Approved,
                ReliabilityLifecycleState::Prepared,
                ReliabilityLifecycleState::Dispatching,
                ReliabilityLifecycleState::Unknown,
            ],
            ReliabilityLifecycleState::Unknown,
        ));

        scenarios.push(run_fake_scenario(
            "paper decision never enters live execution",
            &[
                ReliabilityLifecycleState::Discovered,
                ReliabilityLifecycleState::Evaluated,
                ReliabilityLifecycleState::PaperSimulated,
            ],
            ReliabilityLifecycleState::PaperSimulated,
        ));

        let passed_count = scenarios.iter().filter(|scenario| scenario.passed).count();

        FakeMarketSuiteReport {
            passed: passed_count == scenarios.len(),
            passed_count,
            total_count: scenarios.len(),
            scenarios,
            generated_at: Utc::now(),
        }
    }
}

fn row_to_execution(row: service::sea_orm::QueryResult) -> ExecutionJournalEntry {
    ExecutionJournalEntry {
        id: row.try_get("", "id").unwrap_or_default(),
        decision_id: row.try_get("", "decision_id").unwrap_or_default(),
        item_key: row.try_get("", "item_key").unwrap_or_default(),
        wfm_id: row.try_get("", "wfm_id").unwrap_or_default(),
        side: row.try_get("", "side").unwrap_or_default(),
        operation: row.try_get("", "operation").unwrap_or_default(),
        target_price: row.try_get("", "target_price").unwrap_or_default(),
        target_quantity: row.try_get("", "target_quantity").unwrap_or_default(),
        previous_price: row.try_get("", "previous_price").unwrap_or_default(),
        state: row.try_get("", "state").unwrap_or_default(),
        attempt_count: row.try_get("", "attempt_count").unwrap_or_default(),
        error: row.try_get("", "error").unwrap_or_default(),
        created_at: row.try_get("", "created_at").unwrap_or_default(),
        updated_at: row.try_get("", "updated_at").unwrap_or_default(),
    }
}

fn execution_id(
    decision_id: &str,
    side: TradeSide,
    operation: &str,
    price: i64,
    quantity: i64,
) -> String {
    let key = format!(
        "{decision_id}|{}|{operation}|{price}|{quantity}",
        side_string(side)
    );

    format!("exec:{:016x}", stable_text_hash(&key))
}

// Stable FNV-1a hashing keeps persisted execution IDs deterministic across
// process restarts and Rust toolchain upgrades.
fn stable_text_hash(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;

    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    hash
}

fn run_fake_scenario(
    name: &str,
    states: &[ReliabilityLifecycleState],
    expected: ReliabilityLifecycleState,
) -> FakeMarketScenarioResult {
    let mut current = ReliabilityLifecycleState::Discovered;
    let mut notes = Vec::new();
    let mut passed = true;

    for state in states.iter().copied().skip(1) {
        if transition_allowed(current, state) {
            notes.push(format!(
                "{} -> {} accepted",
                current.as_str(),
                state.as_str()
            ));
            current = state;
        } else {
            notes.push(format!(
                "{} -> {} rejected by state machine",
                current.as_str(),
                state.as_str()
            ));
            passed = false;
            break;
        }
    }

    passed &= current == expected;

    FakeMarketScenarioResult {
        name: name.to_string(),
        passed,
        final_state: current.as_str().to_string(),
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_machine_accepts_normal_trade_path() {
        assert!(transition_allowed(
            ReliabilityLifecycleState::Approved,
            ReliabilityLifecycleState::Prepared
        ));
        assert!(transition_allowed(
            ReliabilityLifecycleState::Prepared,
            ReliabilityLifecycleState::Dispatching
        ));
        assert!(transition_allowed(
            ReliabilityLifecycleState::Dispatching,
            ReliabilityLifecycleState::Active
        ));
        assert!(transition_allowed(
            ReliabilityLifecycleState::Active,
            ReliabilityLifecycleState::Partial
        ));
        assert!(transition_allowed(
            ReliabilityLifecycleState::Partial,
            ReliabilityLifecycleState::Purchased
        ));
    }

    #[test]
    fn state_machine_rejects_backwards_transition() {
        assert!(!transition_allowed(
            ReliabilityLifecycleState::Active,
            ReliabilityLifecycleState::Approved
        ));
    }

    #[test]
    fn execution_id_is_idempotent() {
        let a = execution_id("decision-1", TradeSide::Buy, "upsert", 25, 2);
        let b = execution_id("decision-1", TradeSide::Buy, "upsert", 25, 2);
        let c = execution_id("decision-1", TradeSide::Buy, "upsert", 26, 2);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn fake_market_suite_passes() {
        let report = ReliabilityService::run_fake_market_suite();
        assert!(report.passed);
        assert_eq!(report.passed_count, report.total_count);
    }
}
