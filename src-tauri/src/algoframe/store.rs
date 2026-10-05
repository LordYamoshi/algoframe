use chrono::{DateTime, Duration, Utc};
use service::sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, Statement, TryGetable,
};
use utils::{get_location, Error};

use super::types::*;

fn db_error(component: &str, error: impl ToString) -> Error {
    Error::new(
        format!("AlgoFrame:{component}"),
        error.to_string(),
        get_location!(),
    )
}

fn sql_string(value: &str) -> String {
    value.replace('\'', "''")
}

async fn execute(db: &DatabaseConnection, component: &str, sql: String) -> Result<(), Error> {
    db.execute(Statement::from_string(DatabaseBackend::Sqlite, sql))
        .await
        .map_err(|error| db_error(component, error))?;
    Ok(())
}

pub struct LearningStore;

impl LearningStore {
    pub async fn load_config(db: &DatabaseConnection) -> Result<UltimateConfig, Error> {
        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT value FROM algoframe_setting WHERE key = 'ultimate_config' LIMIT 1"
                    .to_string(),
            ))
            .await
            .map_err(|error| db_error("LoadConfig", error))?;

        if let Some(row) = rows.first() {
            let value: String = row
                .try_get("", "value")
                .map_err(|error| db_error("LoadConfig:Value", error))?;

            if let Ok(config) = serde_json::from_str::<UltimateConfig>(&value) {
                return Ok(config.normalized());
            }
        }

        let config = UltimateConfig::default();
        Self::save_config(db, &config).await?;
        Ok(config)
    }

    pub async fn save_config(
        db: &DatabaseConnection,
        config: &UltimateConfig,
    ) -> Result<(), Error> {
        let value = serde_json::to_string(&config.clone().normalized())
            .map_err(|error| db_error("SaveConfig:Serialize", error))?;

        execute(
            db,
            "SaveConfig",
            format!(
                "INSERT INTO algoframe_setting(key, value, updated_at)
                 VALUES('ultimate_config', '{}', '{}')
                 ON CONFLICT(key) DO UPDATE SET
                   value=excluded.value,
                   updated_at=excluded.updated_at",
                sql_string(&value),
                Utc::now().to_rfc3339(),
            ),
        )
        .await
    }

    pub async fn insert_snapshot(
        db: &DatabaseConnection,
        snapshot: &MarketSnapshot,
    ) -> Result<(), Error> {
        let payload = serde_json::to_string(snapshot)
            .map_err(|error| db_error("InsertSnapshot:Serialize", error))?;

        execute(
            db,
            "InsertSnapshot",
            format!(
                "INSERT OR REPLACE INTO algoframe_market_snapshot(
                    id, item_key, wfm_id, item_name, category, created_at,
                    best_bid, best_ask, mid_price, spread, buy_depth, sell_depth,
                    quality, anomaly_score, regime, granularity, payload
                 ) VALUES(
                    '{}','{}','{}','{}','{}','{}',
                    {},{},{},{},{},{},{},{},'{}','raw','{}'
                 )",
                sql_string(&snapshot.id),
                sql_string(&snapshot.item_key),
                sql_string(&snapshot.wfm_id),
                sql_string(&snapshot.item_name),
                sql_string(&snapshot.category),
                snapshot.created_at.to_rfc3339(),
                snapshot.features.best_bid,
                snapshot.features.best_ask,
                snapshot.features.mid_price,
                snapshot.features.spread,
                snapshot.features.buy_depth,
                snapshot.features.sell_depth,
                snapshot.features.quality.score,
                snapshot.features.anomaly.score,
                snapshot.features.regime.as_str(),
                sql_string(&payload),
            ),
        )
        .await
    }

    pub async fn load_recent_snapshots(
        db: &DatabaseConnection,
        limit: usize,
    ) -> Result<Vec<MarketSnapshot>, Error> {
        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Sqlite,
                format!(
                    "SELECT payload
                     FROM algoframe_market_snapshot
                     WHERE granularity = 'raw'
                     ORDER BY created_at DESC
                     LIMIT {}",
                    limit.max(1)
                ),
            ))
            .await
            .map_err(|error| db_error("LoadSnapshots", error))?;

        let mut snapshots = Vec::new();

        for row in rows {
            let payload: String = row
                .try_get("", "payload")
                .map_err(|error| db_error("LoadSnapshots:Payload", error))?;

            if let Ok(snapshot) = serde_json::from_str::<MarketSnapshot>(&payload) {
                snapshots.push(snapshot);
            }
        }

        snapshots.reverse();
        Ok(snapshots)
    }

    pub async fn count_snapshots(db: &DatabaseConnection) -> Result<usize, Error> {
        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT COUNT(*) AS count FROM algoframe_market_snapshot".to_string(),
            ))
            .await
            .map_err(|error| db_error("CountSnapshots", error))?;

        let count: i64 = rows
            .first()
            .and_then(|row| row.try_get("", "count").ok())
            .unwrap_or(0);

        Ok(count.max(0) as usize)
    }

    pub async fn upsert_decision(
        db: &DatabaseConnection,
        decision: &DecisionRecord,
    ) -> Result<(), Error> {
        let payload = serde_json::to_string(decision)
            .map_err(|error| db_error("UpsertDecision:Serialize", error))?;

        execute(
            db,
            "UpsertDecision",
            format!(
                "INSERT INTO algoframe_decision(
                    id, snapshot_id, item_key, wfm_id, item_name, category, side,
                    status, lifecycle, chosen_action, chosen_propensity,
                    price, quantity, filled_quantity, capital,
                    predicted_profit, predicted_reward,
                    actual_profit, actual_reward,
                    model_version, model_role, policy_name, feature_schema_version,
                    reward_version, policy_version, settings_hash,
                    created_at, updated_at, payload
                 ) VALUES(
                    '{}','{}','{}','{}','{}','{}','{}',
                    '{}','{}','{}',{},
                    {},{},{},{},
                    {},{},
                    {},{},
                    '{}','{:?}','{}',{},{},{},'{}',
                    '{}','{}','{}'
                 )
                 ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    lifecycle=excluded.lifecycle,
                    filled_quantity=excluded.filled_quantity,
                    actual_profit=excluded.actual_profit,
                    actual_reward=excluded.actual_reward,
                    updated_at=excluded.updated_at,
                    payload=excluded.payload",
                sql_string(&decision.id),
                sql_string(&decision.snapshot_id),
                sql_string(&decision.item_key),
                sql_string(&decision.wfm_id),
                sql_string(&decision.item_name),
                sql_string(&decision.category),
                match decision.side {
                    TradeSide::Buy => "buy",
                    TradeSide::Sell => "sell",
                },
                format!("{:?}", decision.status).to_lowercase(),
                format!("{:?}", decision.lifecycle).to_lowercase(),
                sql_string(&decision.chosen_action),
                decision.chosen_propensity,
                decision.price,
                decision.quantity,
                decision.filled_quantity,
                decision.capital,
                decision.predicted_profit,
                decision.predicted_reward,
                decision.actual_profit.unwrap_or(0.0),
                decision.actual_reward.unwrap_or(0.0),
                sql_string(&decision.model_version),
                decision.model_role,
                sql_string(&decision.policy_name),
                decision.feature_schema_version,
                decision.reward_version,
                decision.policy_version,
                sql_string(&decision.settings_hash),
                decision.created_at.to_rfc3339(),
                decision.updated_at.to_rfc3339(),
                sql_string(&payload),
            ),
        )
        .await
    }

    pub async fn load_decisions(
        db: &DatabaseConnection,
        limit: usize,
    ) -> Result<Vec<DecisionRecord>, Error> {
        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Sqlite,
                format!(
                    "SELECT payload FROM algoframe_decision
                     ORDER BY updated_at DESC
                     LIMIT {}",
                    limit.max(1)
                ),
            ))
            .await
            .map_err(|error| db_error("LoadDecisions", error))?;

        let mut decisions = Vec::new();

        for row in rows {
            let payload: String = row
                .try_get("", "payload")
                .map_err(|error| db_error("LoadDecisions:Payload", error))?;

            if let Ok(decision) = serde_json::from_str::<DecisionRecord>(&payload) {
                decisions.push(decision);
            }
        }

        decisions.reverse();
        Ok(decisions)
    }

    pub async fn insert_outcome(
        db: &DatabaseConnection,
        outcome: &OutcomeRecord,
    ) -> Result<(), Error> {
        let payload = serde_json::to_string(outcome)
            .map_err(|error| db_error("InsertOutcome:Serialize", error))?;

        execute(
            db,
            "InsertOutcome",
            format!(
                "INSERT OR REPLACE INTO algoframe_outcome(
                    id, decision_id, item_key, side, outcome_type,
                    quantity, profit, reward, fill_hours, cycle_hours,
                    simulated, created_at, payload
                 ) VALUES(
                    '{}','{}','{}','{}','{}',
                    {},{},{},{},{},{},'{}','{}'
                 )",
                sql_string(&outcome.id),
                sql_string(&outcome.decision_id),
                sql_string(&outcome.item_key),
                match outcome.side {
                    TradeSide::Buy => "buy",
                    TradeSide::Sell => "sell",
                },
                sql_string(&outcome.outcome_type),
                outcome.quantity,
                outcome.profit,
                outcome.reward,
                outcome.fill_hours,
                outcome.cycle_hours,
                if outcome.simulated { 1 } else { 0 },
                outcome.created_at.to_rfc3339(),
                sql_string(&payload),
            ),
        )
        .await
    }

    pub async fn upsert_model(
        db: &DatabaseConnection,
        model: &ModelVersionRecord,
    ) -> Result<(), Error> {
        let payload = serde_json::to_string(model)
            .map_err(|error| db_error("UpsertModel:Serialize", error))?;

        execute(
            db,
            "UpsertModel",
            format!(
                "INSERT INTO algoframe_model_version(
                    id, version, role, active, average_reward, failure_rate,
                    prediction_mae, calibration_error, drawdown,
                    created_at, promoted_at, payload
                 ) VALUES(
                    '{}','{}','{:?}',{},{},{},{},{},{},
                    '{}',{},'{}'
                 )
                 ON CONFLICT(id) DO UPDATE SET
                    role=excluded.role,
                    active=excluded.active,
                    average_reward=excluded.average_reward,
                    failure_rate=excluded.failure_rate,
                    prediction_mae=excluded.prediction_mae,
                    calibration_error=excluded.calibration_error,
                    drawdown=excluded.drawdown,
                    promoted_at=excluded.promoted_at,
                    payload=excluded.payload",
                sql_string(&model.id),
                sql_string(&model.version),
                model.role,
                if model.active { 1 } else { 0 },
                model.average_reward,
                model.failure_rate,
                model.prediction_mae,
                model.calibration_error,
                model.drawdown,
                model.created_at.to_rfc3339(),
                model
                    .promoted_at
                    .map(|value| format!("'{}'", value.to_rfc3339()))
                    .unwrap_or_else(|| "NULL".to_string()),
                sql_string(&payload),
            ),
        )
        .await
    }

    pub async fn insert_evaluation(
        db: &DatabaseConnection,
        policy_name: &str,
        evaluation: &OfflinePolicyEvaluation,
    ) -> Result<(), Error> {
        let id = uuid::Uuid::new_v4().to_string();
        let payload = serde_json::to_string(evaluation)
            .map_err(|error| db_error("InsertEvaluation:Serialize", error))?;

        execute(
            db,
            "InsertEvaluation",
            format!(
                "INSERT INTO algoframe_model_evaluation(
                    id, policy_name, sample_count, average_reward,
                    failure_rate, ips_reward, dr_reward, created_at, payload
                 ) VALUES(
                    '{}','{}',{},{},{},{},{},'{}','{}'
                 )",
                id,
                sql_string(policy_name),
                evaluation.metrics.sample_count,
                evaluation.metrics.average_reward,
                evaluation.metrics.failure_rate,
                evaluation.metrics.ips_reward,
                evaluation.metrics.doubly_robust_reward,
                Utc::now().to_rfc3339(),
                sql_string(&payload),
            ),
        )
        .await
    }

    pub async fn upsert_event(db: &DatabaseConnection, event: &EventSignal) -> Result<(), Error> {
        let payload = serde_json::to_string(event)
            .map_err(|error| db_error("UpsertEvent:Serialize", error))?;

        execute(
            db,
            "UpsertEvent",
            format!(
                "INSERT INTO algoframe_event(
                    id, kind, title, impact, confidence,
                    starts_at, ends_at, source, payload
                 ) VALUES(
                    '{}','{}','{}',{},{},'{}',{},'{}','{}'
                 )
                 ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind,
                    title=excluded.title,
                    impact=excluded.impact,
                    confidence=excluded.confidence,
                    starts_at=excluded.starts_at,
                    ends_at=excluded.ends_at,
                    source=excluded.source,
                    payload=excluded.payload",
                sql_string(&event.id),
                sql_string(&event.kind),
                sql_string(&event.title),
                event.impact,
                event.confidence,
                event.starts_at.to_rfc3339(),
                event
                    .ends_at
                    .map(|value| format!("'{}'", value.to_rfc3339()))
                    .unwrap_or_else(|| "NULL".to_string()),
                sql_string(&event.source),
                sql_string(&payload),
            ),
        )
        .await
    }

    pub async fn load_active_events(db: &DatabaseConnection) -> Result<Vec<EventSignal>, Error> {
        let now = Utc::now().to_rfc3339();

        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Sqlite,
                format!(
                    "SELECT payload FROM algoframe_event
                     WHERE starts_at <= '{now}'
                       AND (ends_at IS NULL OR ends_at >= '{now}')
                     ORDER BY starts_at DESC"
                ),
            ))
            .await
            .map_err(|error| db_error("LoadEvents", error))?;

        let mut events = Vec::new();

        for row in rows {
            let payload: String = row
                .try_get("", "payload")
                .map_err(|error| db_error("LoadEvents:Payload", error))?;

            if let Ok(event) = serde_json::from_str::<EventSignal>(&payload) {
                events.push(event);
            }
        }

        Ok(events)
    }

    pub async fn upsert_graph_edge(db: &DatabaseConnection, edge: &GraphEdge) -> Result<(), Error> {
        let payload = serde_json::to_string(edge)
            .map_err(|error| db_error("UpsertGraphEdge:Serialize", error))?;

        execute(
            db,
            "UpsertGraphEdge",
            format!(
                "INSERT INTO algoframe_graph_edge(
                    id, from_key, to_key, relation, quantity, cost, payload
                 ) VALUES(
                    '{}','{}','{}','{}',{},{},'{}'
                 )
                 ON CONFLICT(id) DO UPDATE SET
                    from_key=excluded.from_key,
                    to_key=excluded.to_key,
                    relation=excluded.relation,
                    quantity=excluded.quantity,
                    cost=excluded.cost,
                    payload=excluded.payload",
                sql_string(&edge.id),
                sql_string(&edge.from_key),
                sql_string(&edge.to_key),
                sql_string(&edge.relation),
                edge.quantity,
                edge.cost,
                sql_string(&payload),
            ),
        )
        .await
    }

    pub async fn load_graph_edges(db: &DatabaseConnection) -> Result<Vec<GraphEdge>, Error> {
        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT payload FROM algoframe_graph_edge".to_string(),
            ))
            .await
            .map_err(|error| db_error("LoadGraphEdges", error))?;

        let mut edges = Vec::new();

        for row in rows {
            let payload: String = row
                .try_get("", "payload")
                .map_err(|error| db_error("LoadGraphEdges:Payload", error))?;

            if let Ok(edge) = serde_json::from_str::<GraphEdge>(&payload) {
                edges.push(edge);
            }
        }

        Ok(edges)
    }

    pub async fn insert_alert(db: &DatabaseConnection, alert: &AlertRecord) -> Result<(), Error> {
        let payload = serde_json::to_string(alert)
            .map_err(|error| db_error("InsertAlert:Serialize", error))?;

        execute(
            db,
            "InsertAlert",
            format!(
                "INSERT OR IGNORE INTO algoframe_alert(
                    id, severity, code, message, item_key,
                    created_at, acknowledged, payload
                 ) VALUES(
                    '{}','{}','{}','{}',{},'{}',{},'{}'
                 )",
                sql_string(&alert.id),
                sql_string(&alert.severity),
                sql_string(&alert.code),
                sql_string(&alert.message),
                alert
                    .item_key
                    .as_ref()
                    .map(|value| format!("'{}'", sql_string(value)))
                    .unwrap_or_else(|| "NULL".to_string()),
                alert.created_at.to_rfc3339(),
                if alert.acknowledged { 1 } else { 0 },
                sql_string(&payload),
            ),
        )
        .await
    }

    pub async fn load_alerts(
        db: &DatabaseConnection,
        limit: usize,
    ) -> Result<Vec<AlertRecord>, Error> {
        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Sqlite,
                format!(
                    "SELECT payload FROM algoframe_alert
                     ORDER BY created_at DESC LIMIT {}",
                    limit.max(1)
                ),
            ))
            .await
            .map_err(|error| db_error("LoadAlerts", error))?;

        let mut alerts = Vec::new();

        for row in rows {
            let payload: String = row
                .try_get("", "payload")
                .map_err(|error| db_error("LoadAlerts:Payload", error))?;

            if let Ok(alert) = serde_json::from_str::<AlertRecord>(&payload) {
                alerts.push(alert);
            }
        }

        Ok(alerts)
    }

    pub async fn write_inspector(
        db: &DatabaseConnection,
        inspector: &LearningInspector,
    ) -> Result<(), Error> {
        let payload = serde_json::to_string_pretty(inspector)
            .map_err(|error| db_error("WriteInspector:Serialize", error))?;

        execute(
            db,
            "WriteInspector",
            format!(
                "INSERT INTO algoframe_setting(key, value, updated_at)
                 VALUES('latest_inspector','{}','{}')
                 ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at",
                sql_string(&payload),
                Utc::now().to_rfc3339(),
            ),
        )
        .await
    }

    pub async fn load_inspector(db: &DatabaseConnection) -> Result<LearningInspector, Error> {
        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Sqlite,
                "SELECT value FROM algoframe_setting
                 WHERE key='latest_inspector' LIMIT 1"
                    .to_string(),
            ))
            .await
            .map_err(|error| db_error("LoadInspector", error))?;

        if let Some(row) = rows.first() {
            let value: String = row
                .try_get("", "value")
                .map_err(|error| db_error("LoadInspector:Value", error))?;

            if let Ok(inspector) = serde_json::from_str::<LearningInspector>(&value) {
                return Ok(inspector);
            }
        }

        Ok(LearningInspector::default())
    }

    pub async fn forget_item(db: &DatabaseConnection, item_key: &str) -> Result<(), Error> {
        let item_key = sql_string(item_key);

        for table in [
            "algoframe_market_snapshot",
            "algoframe_decision",
            "algoframe_outcome",
        ] {
            execute(
                db,
                "ForgetItem",
                format!("DELETE FROM {table} WHERE item_key='{item_key}'"),
            )
            .await?;
        }

        Ok(())
    }

    pub async fn forget_category(db: &DatabaseConnection, category: &str) -> Result<(), Error> {
        let category = sql_string(category);

        // Outcomes do not duplicate category, so remove them through the
        // decision relation before removing the decisions themselves.
        execute(
            db,
            "ForgetCategory:Outcomes",
            format!(
                "DELETE FROM algoframe_outcome
                 WHERE decision_id IN (
                    SELECT id FROM algoframe_decision
                    WHERE category='{category}'
                 )"
            ),
        )
        .await?;

        execute(
            db,
            "ForgetCategory:Decisions",
            format!(
                "DELETE FROM algoframe_decision
                 WHERE category='{category}'"
            ),
        )
        .await?;

        execute(
            db,
            "ForgetCategory:Snapshots",
            format!(
                "DELETE FROM algoframe_market_snapshot
                 WHERE category='{category}'"
            ),
        )
        .await?;

        Ok(())
    }

    pub async fn reset_learning(
        db: &DatabaseConnection,
        keep_snapshots: bool,
    ) -> Result<(), Error> {
        for table in [
            "algoframe_decision",
            "algoframe_outcome",
            "algoframe_model_version",
            "algoframe_model_evaluation",
            "algoframe_alert",
        ] {
            execute(db, "ResetLearning", format!("DELETE FROM {table}")).await?;
        }

        if !keep_snapshots {
            execute(
                db,
                "ResetLearning:Snapshots",
                "DELETE FROM algoframe_market_snapshot".to_string(),
            )
            .await?;
        }

        execute(
            db,
            "ResetLearning:Inspector",
            "DELETE FROM algoframe_setting WHERE key='latest_inspector'".to_string(),
        )
        .await?;

        Ok(())
    }

    pub async fn forget_before(
        db: &DatabaseConnection,
        before: DateTime<Utc>,
    ) -> Result<(), Error> {
        let before = before.to_rfc3339();

        for table in [
            "algoframe_market_snapshot",
            "algoframe_decision",
            "algoframe_outcome",
            "algoframe_model_evaluation",
            "algoframe_alert",
        ] {
            execute(
                db,
                "ForgetBefore",
                format!("DELETE FROM {table} WHERE created_at < '{before}'"),
            )
            .await?;
        }

        Ok(())
    }

    pub async fn run_snapshot_retention(
        db: &DatabaseConnection,
        config: &UltimateConfig,
    ) -> Result<(), Error> {
        let now = Utc::now();
        let five_cutoff = (now - Duration::days(config.snapshot_full_days.max(1))).to_rfc3339();
        let thirty_cutoff = (now
            - Duration::days(
                config
                    .snapshot_five_min_days
                    .max(config.snapshot_full_days + 1),
            ))
        .to_rfc3339();
        let hourly_cutoff = (now
            - Duration::days(
                config
                    .snapshot_thirty_min_days
                    .max(config.snapshot_five_min_days + 1),
            ))
        .to_rfc3339();
        let delete_cutoff = (now
            - Duration::days(
                config
                    .snapshot_hourly_days
                    .max(config.snapshot_thirty_min_days + 1),
            ))
        .to_rfc3339();

        // SQLite JSON aggregation is avoided here: aggregation uses numeric columns
        // already stored alongside the payload, then preserves a compact JSON summary.
        for (source, target, cutoff, bucket_seconds) in [
            ("raw", "5m", &five_cutoff, 300_i64),
            ("5m", "30m", &thirty_cutoff, 1800_i64),
            ("30m", "1h", &hourly_cutoff, 3600_i64),
        ] {
            let aggregate_sql = format!(
                "INSERT OR IGNORE INTO algoframe_market_snapshot(
                    id, item_key, wfm_id, item_name, category, created_at,
                    best_bid, best_ask, mid_price, spread,
                    buy_depth, sell_depth, quality, anomaly_score,
                    regime, granularity, payload
                 )
                 SELECT
                    item_key || ':' || '{target}' || ':' ||
                        CAST((unixepoch(created_at) / {bucket_seconds}) AS TEXT),
                    item_key,
                    MIN(wfm_id),
                    MIN(item_name),
                    MIN(category),
                    strftime(
                        '%Y-%m-%dT%H:%M:%fZ',
                        (unixepoch(created_at) / {bucket_seconds}) * {bucket_seconds},
                        'unixepoch'
                    ),
                    CAST(AVG(best_bid) AS INTEGER),
                    CAST(AVG(best_ask) AS INTEGER),
                    AVG(mid_price),
                    AVG(spread),
                    CAST(AVG(buy_depth) AS INTEGER),
                    CAST(AVG(sell_depth) AS INTEGER),
                    AVG(quality),
                    AVG(anomaly_score),
                    MIN(regime),
                    '{target}',
                    json_object(
                        'aggregated', 1,
                        'source', '{source}',
                        'samples', COUNT(*),
                        'avg_mid', AVG(mid_price),
                        'avg_spread', AVG(spread),
                        'avg_quality', AVG(quality),
                        'avg_anomaly', AVG(anomaly_score)
                    )
                 FROM algoframe_market_snapshot
                 WHERE granularity='{source}'
                   AND created_at < '{cutoff}'
                 GROUP BY item_key, (unixepoch(created_at) / {bucket_seconds})"
            );

            execute(db, "SnapshotRetention:Aggregate", aggregate_sql).await?;

            execute(
                db,
                "SnapshotRetention:DeleteSource",
                format!(
                    "DELETE FROM algoframe_market_snapshot
                     WHERE granularity='{source}'
                       AND created_at < '{cutoff}'"
                ),
            )
            .await?;
        }

        execute(
            db,
            "SnapshotRetention:DeleteOld",
            format!(
                "DELETE FROM algoframe_market_snapshot
                 WHERE granularity='1h'
                   AND created_at < '{delete_cutoff}'"
            ),
        )
        .await?;

        Ok(())
    }
}
