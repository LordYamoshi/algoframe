use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utils::{get_location, Error};

use crate::{
    algoframe::{
        events::refresh_event_intelligence,
        math::{normalized_reward, offline_policy_evaluation},
        EventSignal, GraphEdge, LearningInspector, LearningStore, OperatingMode, TradeSide,
        UltimateConfig,
    },
    DATABASE,
};

#[tauri::command]
pub async fn learning_get_inspector() -> Result<LearningInspector, Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:GetInspector",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    LearningStore::load_inspector(db).await
}

#[tauri::command]
pub async fn learning_get_config() -> Result<UltimateConfig, Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:GetConfig",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    LearningStore::load_config(db).await
}

#[tauri::command]
pub async fn learning_update_config(config: UltimateConfig) -> Result<UltimateConfig, Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:UpdateConfig",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    let mut config = config.normalized();
    config.settings_revision = config.settings_revision.saturating_add(1);

    LearningStore::save_config(db, &config).await?;
    Ok(config)
}

#[tauri::command]
pub async fn learning_set_mode(mode: OperatingMode) -> Result<UltimateConfig, Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:SetMode",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    let mut config = LearningStore::load_config(db).await?;
    config.mode = mode;
    config.settings_revision = config.settings_revision.saturating_add(1);
    LearningStore::save_config(db, &config).await?;
    Ok(config)
}

#[tauri::command]
pub async fn learning_reset(keep_snapshots: bool) -> Result<(), Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:Reset",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    LearningStore::reset_learning(db, keep_snapshots).await
}

#[tauri::command]
pub async fn learning_forget_item(item_key: String) -> Result<(), Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:ForgetItem",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    LearningStore::forget_item(db, &item_key).await
}

#[tauri::command]
pub async fn learning_forget_category(category: String) -> Result<(), Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:ForgetCategory",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    LearningStore::forget_category(db, &category).await
}

#[tauri::command]
pub async fn learning_forget_before(before_rfc3339: String) -> Result<(), Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:ForgetBefore",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    let before = DateTime::parse_from_rfc3339(&before_rfc3339)
        .map_err(|error| {
            Error::new(
                "AlgoFrame:Learning:ForgetBefore:Parse",
                error.to_string(),
                get_location!(),
            )
        })?
        .with_timezone(&Utc);

    LearningStore::forget_before(db, before).await
}

#[tauri::command]
pub async fn learning_upsert_event(event: EventSignal) -> Result<(), Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:UpsertEvent",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    LearningStore::upsert_event(db, &event).await
}

#[tauri::command]
pub async fn learning_refresh_events() -> Result<Value, Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:RefreshEvents",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    let events = refresh_event_intelligence(db).await?;

    serde_json::to_value(events).map_err(|error| {
        Error::new(
            "AlgoFrame:Learning:RefreshEvents:Serialize",
            error.to_string(),
            get_location!(),
        )
    })
}

#[tauri::command]
pub async fn learning_upsert_graph_edge(edge: GraphEdge) -> Result<(), Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:UpsertGraphEdge",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    LearningStore::upsert_graph_edge(db, &edge).await
}

#[tauri::command]
pub async fn learning_get_decisions(limit: usize) -> Result<Value, Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:GetDecisions",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    let decisions = LearningStore::load_decisions(db, limit.clamp(1, 10_000)).await?;

    serde_json::to_value(decisions).map_err(|error| {
        Error::new(
            "AlgoFrame:Learning:GetDecisions:Serialize",
            error.to_string(),
            get_location!(),
        )
    })
}

#[tauri::command]
pub async fn learning_replay_decision(decision_id: String) -> Result<Value, Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:Replay",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    let decisions = LearningStore::load_decisions(db, 25_000).await?;

    let decision = decisions
        .into_iter()
        .find(|decision| decision.id == decision_id)
        .ok_or_else(|| {
            Error::new(
                "AlgoFrame:Learning:Replay",
                "Decision was not found",
                get_location!(),
            )
        })?;

    let snapshots = LearningStore::load_recent_snapshots(db, 8_000).await?;
    let snapshot = snapshots
        .into_iter()
        .find(|snapshot| snapshot.id == decision.snapshot_id);

    Ok(serde_json::json!({
        "decision": decision.clone(),
        "snapshot": snapshot,
        "replay": {
            "deterministic_seed": decision.seed,
            "settings_hash": decision.settings_hash.clone(),
            "model_version": decision.model_version.clone(),
            "feature_schema_version": decision.feature_schema_version,
            "reward_version": decision.reward_version,
            "policy_version": decision.policy_version,
            "propensities": decision.propensities.clone(),
            "shadow_actions": decision.shadow_actions.clone(),
        }
    }))
}

#[tauri::command]
pub async fn learning_counterfactual_replay(decision_id: String) -> Result<Value, Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:Counterfactual",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    let config = LearningStore::load_config(db).await?;
    let decisions = LearningStore::load_decisions(db, 25_000).await?;

    let decision = decisions
        .iter()
        .find(|decision| decision.id == decision_id)
        .cloned()
        .ok_or_else(|| {
            Error::new(
                "AlgoFrame:Learning:Counterfactual",
                "Decision was not found",
                get_location!(),
            )
        })?;

    let snapshots = LearningStore::load_recent_snapshots(db, 50_000).await?;
    let horizon_end = decision.created_at + chrono::Duration::hours(24);

    let mut relevant: Vec<_> = snapshots
        .into_iter()
        .filter(|snapshot| {
            snapshot.item_key == decision.item_key
                && snapshot.created_at >= decision.created_at
                && snapshot.created_at <= horizon_end
        })
        .collect();

    relevant.sort_by_key(|snapshot| snapshot.created_at);

    if relevant.is_empty() {
        return Ok(serde_json::json!({
            "decision": decision,
            "alternatives": [],
            "note": "No recorded future market snapshots are available for this decision."
        }));
    }

    let reference_bid = decision.features.robust_bid.max(1);
    let reference_ask = decision.features.robust_ask.max(reference_bid + 1);
    let max_steps = config.max_price_search_steps.max(2);
    let max_quantity = config
        .max_quantity_search
        .min(config.max_trade_quantity)
        .max(1);

    let prices: Vec<i64> = match decision.side {
        TradeSide::Buy => {
            let lower = reference_bid.saturating_sub(2).max(1);
            let upper = reference_bid.saturating_add(max_steps);
            (lower..=upper).collect()
        }
        TradeSide::Sell => {
            let lower = reference_ask.saturating_sub(max_steps).max(1);
            let upper = reference_ask.saturating_add(3);
            (lower..=upper).collect()
        }
    };

    let unit_cost = if decision.quantity > 0 {
        decision.capital / decision.quantity as f64
    } else {
        decision.price.max(1) as f64
    };

    let mut alternatives = Vec::new();

    for candidate_price in prices {
        for quantity in 1..=max_quantity {
            let fill = relevant.iter().find(|snapshot| match decision.side {
                TradeSide::Buy => {
                    snapshot.features.robust_ask > 0
                        && snapshot.features.robust_ask <= candidate_price
                }
                TradeSide::Sell => snapshot.features.robust_bid >= candidate_price,
            });

            let Some(fill_snapshot) = fill else {
                alternatives.push(serde_json::json!({
                    "price": candidate_price,
                    "quantity": quantity,
                    "filled": false,
                    "fill_hours": null,
                    "cycle_hours": 24.0,
                    "profit": 0.0,
                    "reward": -0.02,
                }));
                continue;
            };

            let fill_hours = (fill_snapshot.created_at - decision.created_at)
                .num_seconds()
                .max(0) as f64
                / 3600.0;

            match decision.side {
                TradeSide::Sell => {
                    let profit_per_unit = (candidate_price as f64 - unit_cost).max(0.0);
                    let capital = unit_cost.max(1.0) * quantity as f64;

                    let reward = normalized_reward(
                        profit_per_unit * quantity as f64,
                        capital,
                        fill_hours.max(0.05),
                        config.human_minutes_per_trade,
                        config.human_time_value_plat_per_hour,
                    );

                    alternatives.push(serde_json::json!({
                        "price": candidate_price,
                        "quantity": quantity,
                        "filled": true,
                        "fill_hours": fill_hours,
                        "cycle_hours": fill_hours,
                        "profit": profit_per_unit,
                        "reward": reward,
                    }));
                }
                TradeSide::Buy => {
                    let target_sell = candidate_price as f64 + decision.predicted_profit.max(1.0);

                    let sale = relevant.iter().find(|snapshot| {
                        snapshot.created_at >= fill_snapshot.created_at
                            && snapshot.features.robust_bid as f64 >= target_sell
                    });

                    if let Some(sale_snapshot) = sale {
                        let cycle_hours = (sale_snapshot.created_at - decision.created_at)
                            .num_seconds()
                            .max(0) as f64
                            / 3600.0;

                        let profit_per_unit =
                            sale_snapshot.features.robust_bid as f64 - candidate_price as f64;

                        let capital = candidate_price.max(1) as f64 * quantity as f64;

                        let reward = normalized_reward(
                            profit_per_unit * quantity as f64,
                            capital,
                            cycle_hours.max(0.05),
                            config.human_minutes_per_trade,
                            config.human_time_value_plat_per_hour,
                        );

                        alternatives.push(serde_json::json!({
                            "price": candidate_price,
                            "quantity": quantity,
                            "filled": true,
                            "sold": true,
                            "fill_hours": fill_hours,
                            "cycle_hours": cycle_hours,
                            "profit": profit_per_unit,
                            "reward": reward,
                        }));
                    } else {
                        alternatives.push(serde_json::json!({
                            "price": candidate_price,
                            "quantity": quantity,
                            "filled": true,
                            "sold": false,
                            "fill_hours": fill_hours,
                            "cycle_hours": 24.0,
                            "profit": 0.0,
                            "reward": -0.01,
                        }));
                    }
                }
            }
        }
    }

    alternatives.sort_by(|a, b| {
        let ar = a.get("reward").and_then(Value::as_f64).unwrap_or(f64::MIN);
        let br = b.get("reward").and_then(Value::as_f64).unwrap_or(f64::MIN);
        br.partial_cmp(&ar).unwrap_or(std::cmp::Ordering::Equal)
    });

    let best_alternative = alternatives.first().cloned();

    Ok(serde_json::json!({
        "decision": decision,
        "recorded_snapshot_count": relevant.len(),
        "horizon_hours": 24,
        "alternatives": alternatives,
        "best_alternative": best_alternative,
        "note": "Digital-twin replay is conservative: it uses recorded external order-book crossings and does not assume queue priority.",
    }))
}

#[tauri::command]
pub async fn learning_run_offline_evaluation() -> Result<Value, Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:Evaluate",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    let decisions = LearningStore::load_decisions(db, 25_000).await?;

    let policies = [
        "shadow_conservative",
        "shadow_aggressive",
        "shadow_fast_turnover",
        "shadow_max_reward",
        "shadow_balanced",
    ];

    let evaluations: Vec<_> = policies
        .iter()
        .map(|policy| offline_policy_evaluation(&decisions, policy))
        .collect();

    serde_json::to_value(evaluations).map_err(|error| {
        Error::new(
            "AlgoFrame:Learning:Evaluate:Serialize",
            error.to_string(),
            get_location!(),
        )
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningExportResult {
    pub path: String,
    pub decision_count: usize,
    pub snapshot_count: usize,
}

#[tauri::command]
pub async fn learning_export_dataset() -> Result<LearningExportResult, Error> {
    let db = DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Learning:Export",
            "Database is not initialized",
            get_location!(),
        )
    })?;

    let decisions = LearningStore::load_decisions(db, 50_000).await?;
    let snapshots = LearningStore::load_recent_snapshots(db, 50_000).await?;
    let inspector = LearningStore::load_inspector(db).await?;
    let config = LearningStore::load_config(db).await?;

    let export = serde_json::json!({
        "exported_at": Utc::now(),
        "config": config,
        "inspector": inspector,
        "decisions": decisions,
        "snapshots": snapshots,
    });

    let path = std::path::PathBuf::from(utils::get_base_path()).join(format!(
        "algoframe_learning_export_{}.json",
        Utc::now().format("%Y%m%d_%H%M%S")
    ));

    std::fs::write(
        &path,
        serde_json::to_string_pretty(&export).map_err(|error| {
            Error::new(
                "AlgoFrame:Learning:Export:Serialize",
                error.to_string(),
                get_location!(),
            )
        })?,
    )
    .map_err(|error| {
        Error::new(
            "AlgoFrame:Learning:Export:Write",
            error.to_string(),
            get_location!(),
        )
    })?;

    Ok(LearningExportResult {
        path: path.to_string_lossy().to_string(),
        decision_count: export["decisions"].as_array().map(|v| v.len()).unwrap_or(0),
        snapshot_count: export["snapshots"].as_array().map(|v| v.len()).unwrap_or(0),
    })
}
