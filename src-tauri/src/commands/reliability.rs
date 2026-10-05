use serde_json::Value;
use service::sea_orm::ConnectionTrait;
use utils::{get_location, Error};

use crate::{
    algoframe::{
        FakeMarketSuiteReport, ReleaseGateReport, ReliabilityConfig, ReliabilityService,
        ReliabilityStatus,
    },
    DATABASE,
};

fn db() -> Result<&'static service::sea_orm::DatabaseConnection, Error> {
    DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Reliability:Command",
            "Database is not initialized",
            get_location!(),
        )
    })
}

#[tauri::command]
pub async fn reliability_get_status() -> Result<ReliabilityStatus, Error> {
    ReliabilityService::get_status(db()?).await
}

#[tauri::command]
pub async fn reliability_update_config(
    config: ReliabilityConfig,
) -> Result<ReliabilityConfig, Error> {
    let config = config.normalized();
    ReliabilityService::save_config(db()?, &config).await?;
    Ok(config)
}

#[tauri::command]
pub async fn reliability_trip_circuit(reason: String) -> Result<(), Error> {
    let reason = if reason.trim().is_empty() {
        "manual safety freeze".to_string()
    } else {
        reason.trim().to_string()
    };

    ReliabilityService::trip_manual(db()?, &reason).await
}

#[tauri::command]
pub async fn reliability_clear_circuit() -> Result<(), Error> {
    ReliabilityService::clear_manual(db()?).await
}

#[tauri::command]
pub async fn reliability_resolve_execution(
    execution_id: String,
    resolution: String,
) -> Result<(), Error> {
    ReliabilityService::resolve_execution(db()?, &execution_id, &resolution).await
}

#[tauri::command]
pub async fn reliability_run_fake_market_suite() -> Result<FakeMarketSuiteReport, Error> {
    Ok(ReliabilityService::run_fake_market_suite())
}

#[tauri::command]
pub async fn reliability_run_release_gate(
    candidate_policy: Option<String>,
) -> Result<ReleaseGateReport, Error> {
    ReliabilityService::run_release_gate(db()?, candidate_policy.as_deref()).await
}

#[tauri::command]
pub async fn reliability_get_release_gate_history(
    limit: Option<usize>,
) -> Result<Vec<Value>, Error> {
    let db = db()?;
    let rows = db
        .query_all(service::sea_orm::Statement::from_string(
            service::sea_orm::DatabaseBackend::Sqlite,
            format!(
                "SELECT payload
                 FROM algoframe_release_gate
                 ORDER BY created_at DESC
                 LIMIT {}",
                limit.unwrap_or(25).clamp(1, 500)
            ),
        ))
        .await
        .map_err(|error| {
            Error::new(
                "AlgoFrame:Reliability:ReleaseGateHistory",
                error.to_string(),
                get_location!(),
            )
        })?;

    Ok(rows
        .into_iter()
        .filter_map(|row| {
            use service::sea_orm::TryGetable;
            let payload: String = row.try_get("", "payload").ok()?;
            serde_json::from_str::<Value>(&payload).ok()
        })
        .collect())
}
