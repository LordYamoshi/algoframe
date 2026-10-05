use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use service::sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, Statement, TryGetable,
};
use std::{fs, path::PathBuf};
use utils::{get_location, Error};

use crate::{
    algoframe::{LearningStore, OperatingMode, UltimateConfig},
    helper, DATABASE,
};

fn db() -> Result<&'static DatabaseConnection, Error> {
    DATABASE.get().ok_or_else(|| {
        Error::new(
            "AlgoFrame:Product:Database",
            "Database is not initialized",
            get_location!(),
        )
    })
}

fn product_error(component: &str, error: impl ToString) -> Error {
    Error::new(
        format!("AlgoFrame:Product:{component}"),
        error.to_string(),
        get_location!(),
    )
}

fn sql_string(value: &str) -> String {
    value.replace('\'', "''")
}

async fn query_i64(
    db: &DatabaseConnection,
    sql: impl Into<String>,
    column: &str,
) -> Result<i64, Error> {
    let row = db
        .query_one(Statement::from_string(DatabaseBackend::Sqlite, sql.into()))
        .await
        .map_err(|error| product_error("QueryI64", error))?;

    Ok(row
        .and_then(|row| row.try_get("", column).ok())
        .unwrap_or(0_i64))
}

async fn query_string(
    db: &DatabaseConnection,
    sql: impl Into<String>,
    column: &str,
) -> Result<String, Error> {
    let row = db
        .query_one(Statement::from_string(DatabaseBackend::Sqlite, sql.into()))
        .await
        .map_err(|error| product_error("QueryString", error))?;

    Ok(row
        .and_then(|row| row.try_get("", column).ok())
        .unwrap_or_default())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductHealth {
    pub checked_at: DateTime<Utc>,
    pub mode: OperatingMode,

    pub database_integrity: String,
    pub database_size_bytes: i64,
    pub snapshot_count: i64,
    pub decision_count: i64,
    pub outcome_count: i64,
    pub alert_count: i64,

    pub latest_snapshot_at: Option<DateTime<Utc>>,
    pub latest_decision_at: Option<DateTime<Utc>>,
    pub snapshot_age_minutes: Option<f64>,

    pub model_health_score: f64,
    pub model_fallback_active: bool,
    pub model_healthy: bool,

    pub degraded: bool,
    pub degraded_reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductBackup {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductProfile {
    pub name: String,
    pub config: UltimateConfig,
    pub updated_at: DateTime<Utc>,
}

fn parse_rfc3339(value: &str) -> Option<DateTime<Utc>> {
    if value.trim().is_empty() {
        return None;
    }

    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|value| value.with_timezone(&Utc))
}

async fn main_database_path(db: &DatabaseConnection) -> Result<PathBuf, Error> {
    let rows = db
        .query_all(Statement::from_string(
            DatabaseBackend::Sqlite,
            "PRAGMA database_list".to_string(),
        ))
        .await
        .map_err(|error| product_error("DatabasePath", error))?;

    for row in rows {
        let name: String = row.try_get("", "name").unwrap_or_default();
        let file: String = row.try_get("", "file").unwrap_or_default();

        if name == "main" && !file.is_empty() {
            return Ok(PathBuf::from(file));
        }
    }

    Err(product_error(
        "DatabasePath",
        "SQLite main database path was unavailable",
    ))
}

#[tauri::command]
pub async fn algoframe_product_health() -> Result<ProductHealth, Error> {
    let db = db()?;
    let inspector = LearningStore::load_inspector(db).await?;

    let integrity = query_string(db, "PRAGMA integrity_check", "integrity_check").await?;
    let page_count = query_i64(db, "PRAGMA page_count", "page_count").await?;
    let page_size = query_i64(db, "PRAGMA page_size", "page_size").await?;

    let snapshot_count = query_i64(
        db,
        "SELECT COUNT(*) AS count FROM algoframe_market_snapshot",
        "count",
    )
    .await?;

    let decision_count = query_i64(
        db,
        "SELECT COUNT(*) AS count FROM algoframe_decision",
        "count",
    )
    .await?;

    let outcome_count = query_i64(
        db,
        "SELECT COUNT(*) AS count FROM algoframe_outcome",
        "count",
    )
    .await?;

    let alert_count = query_i64(
        db,
        "SELECT COUNT(*) AS count FROM algoframe_alert WHERE acknowledged = 0",
        "count",
    )
    .await?;

    let latest_snapshot_raw = query_string(
        db,
        "SELECT COALESCE(MAX(created_at), '') AS latest FROM algoframe_market_snapshot",
        "latest",
    )
    .await?;

    let latest_decision_raw = query_string(
        db,
        "SELECT COALESCE(MAX(updated_at), '') AS latest FROM algoframe_decision",
        "latest",
    )
    .await?;

    let latest_snapshot_at = parse_rfc3339(&latest_snapshot_raw);
    let latest_decision_at = parse_rfc3339(&latest_decision_raw);

    let snapshot_age_minutes =
        latest_snapshot_at.map(|time| (Utc::now() - time).num_seconds().max(0) as f64 / 60.0);

    let mut degraded_reasons = Vec::new();

    if integrity.to_lowercase() != "ok" {
        degraded_reasons.push(format!("SQLite integrity check returned '{integrity}'"));
    }

    if inspector.health.fallback_active {
        degraded_reasons.push("Learning engine is using conservative fallback".to_string());
    }

    if !inspector.health.healthy {
        degraded_reasons.extend(inspector.health.reasons.iter().cloned());
    }

    if snapshot_count > 0 && snapshot_age_minutes.unwrap_or(0.0) > 30.0 {
        degraded_reasons.push(format!(
            "Market snapshots are {:.0} minutes old",
            snapshot_age_minutes.unwrap_or_default()
        ));
    }

    if inspector.health.score < 0.50 {
        degraded_reasons.push(format!(
            "Model health is low ({:.0}%)",
            inspector.health.score * 100.0
        ));
    }

    Ok(ProductHealth {
        checked_at: Utc::now(),
        mode: inspector.mode,
        database_integrity: integrity,
        database_size_bytes: page_count.saturating_mul(page_size),
        snapshot_count,
        decision_count,
        outcome_count,
        alert_count,
        latest_snapshot_at,
        latest_decision_at,
        snapshot_age_minutes,
        model_health_score: inspector.health.score,
        model_fallback_active: inspector.health.fallback_active,
        model_healthy: inspector.health.healthy,
        degraded: !degraded_reasons.is_empty(),
        degraded_reasons,
    })
}

#[tauri::command]
pub async fn algoframe_product_backup_database() -> Result<ProductBackup, Error> {
    let db = db()?;
    let database_path = main_database_path(db).await?;

    let backup_dir = helper::get_app_storage_path().join("algoframe_backups");
    fs::create_dir_all(&backup_dir)
        .map_err(|error| product_error("Backup:CreateDirectory", error))?;

    let name = format!(
        "algoframe_backup_{}.sqlite",
        Utc::now().format("%Y%m%d_%H%M%S")
    );

    let backup_path = backup_dir.join(&name);
    let backup_sql_path = sql_string(&backup_path.to_string_lossy());

    db.execute(Statement::from_string(
        DatabaseBackend::Sqlite,
        format!("VACUUM INTO '{backup_sql_path}'"),
    ))
    .await
    .map_err(|error| product_error("Backup:VacuumInto", error))?;

    let metadata =
        fs::metadata(&backup_path).map_err(|error| product_error("Backup:Metadata", error))?;

    let created_at = metadata
        .created()
        .ok()
        .map(DateTime::<Utc>::from)
        .unwrap_or_else(Utc::now);

    // Make sure the source path was valid too. This is useful diagnostic
    // context if a future SQLite configuration changes.
    if !database_path.exists() {
        return Err(product_error(
            "Backup:Source",
            "Database backup succeeded but the main database path is no longer present",
        ));
    }

    Ok(ProductBackup {
        name,
        path: backup_path.to_string_lossy().to_string(),
        size_bytes: metadata.len(),
        created_at,
    })
}

#[tauri::command]
pub async fn algoframe_product_list_backups() -> Result<Vec<ProductBackup>, Error> {
    let backup_dir = helper::get_app_storage_path().join("algoframe_backups");

    if !backup_dir.exists() {
        return Ok(vec![]);
    }

    let mut backups = Vec::new();

    for entry in fs::read_dir(&backup_dir).map_err(|error| product_error("Backup:List", error))? {
        let entry = entry.map_err(|error| product_error("Backup:ListEntry", error))?;
        let path = entry.path();

        if path.extension().and_then(|value| value.to_str()) != Some("sqlite") {
            continue;
        }

        let metadata = entry
            .metadata()
            .map_err(|error| product_error("Backup:Metadata", error))?;

        let created_at = metadata
            .created()
            .ok()
            .map(DateTime::<Utc>::from)
            .unwrap_or_else(Utc::now);

        backups.push(ProductBackup {
            name: entry.file_name().to_string_lossy().to_string(),
            path: path.to_string_lossy().to_string(),
            size_bytes: metadata.len(),
            created_at,
        });
    }

    backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(backups)
}

#[tauri::command]
pub async fn algoframe_product_vacuum_database() -> Result<(), Error> {
    let db = db()?;

    db.execute(Statement::from_string(
        DatabaseBackend::Sqlite,
        "VACUUM".to_string(),
    ))
    .await
    .map_err(|error| product_error("Vacuum", error))?;

    Ok(())
}

fn validate_profile_name(name: &str) -> Result<String, Error> {
    let clean = name.trim();

    if clean.is_empty() {
        return Err(product_error("Profile", "Profile name cannot be empty"));
    }

    if clean.len() > 60 {
        return Err(product_error(
            "Profile",
            "Profile name must be 60 characters or fewer",
        ));
    }

    if clean.chars().any(|ch| ch.is_control()) {
        return Err(product_error(
            "Profile",
            "Profile name contains invalid control characters",
        ));
    }

    Ok(clean.to_string())
}

#[tauri::command]
pub async fn algoframe_product_save_profile(
    name: String,
    config: UltimateConfig,
) -> Result<ProductProfile, Error> {
    let db = db()?;
    let name = validate_profile_name(&name)?;
    let now = Utc::now();

    let profile = ProductProfile {
        name: name.clone(),
        config: config.normalized(),
        updated_at: now,
    };

    let value = serde_json::to_string(&profile)
        .map_err(|error| product_error("Profile:Serialize", error))?;

    let key = format!("product_profile:{name}");

    db.execute(Statement::from_string(
        DatabaseBackend::Sqlite,
        format!(
            "INSERT INTO algoframe_setting(key, value, updated_at)
             VALUES('{}', '{}', '{}')
             ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at",
            sql_string(&key),
            sql_string(&value),
            now.to_rfc3339(),
        ),
    ))
    .await
    .map_err(|error| product_error("Profile:Save", error))?;

    Ok(profile)
}

#[tauri::command]
pub async fn algoframe_product_list_profiles() -> Result<Vec<ProductProfile>, Error> {
    let db = db()?;

    let rows = db
        .query_all(Statement::from_string(
            DatabaseBackend::Sqlite,
            "SELECT value
             FROM algoframe_setting
             WHERE key LIKE 'product_profile:%'
             ORDER BY updated_at DESC"
                .to_string(),
        ))
        .await
        .map_err(|error| product_error("Profile:List", error))?;

    let mut profiles = Vec::new();

    for row in rows {
        let value: String = row.try_get("", "value").unwrap_or_default();

        if let Ok(profile) = serde_json::from_str::<ProductProfile>(&value) {
            profiles.push(profile);
        }
    }

    Ok(profiles)
}

async fn load_profile(db: &DatabaseConnection, name: &str) -> Result<ProductProfile, Error> {
    let name = validate_profile_name(name)?;
    let key = format!("product_profile:{name}");

    let row = db
        .query_one(Statement::from_string(
            DatabaseBackend::Sqlite,
            format!(
                "SELECT value
                 FROM algoframe_setting
                 WHERE key='{}'
                 LIMIT 1",
                sql_string(&key)
            ),
        ))
        .await
        .map_err(|error| product_error("Profile:Load", error))?
        .ok_or_else(|| product_error("Profile:Load", "Profile was not found"))?;

    let value: String = row
        .try_get("", "value")
        .map_err(|error| product_error("Profile:LoadValue", error))?;

    serde_json::from_str(&value).map_err(|error| product_error("Profile:Deserialize", error))
}

#[tauri::command]
pub async fn algoframe_product_apply_profile(name: String) -> Result<UltimateConfig, Error> {
    let db = db()?;
    let profile = load_profile(db, &name).await?;

    let mut config = profile.config.normalized();
    config.settings_revision = config.settings_revision.saturating_add(1);

    LearningStore::save_config(db, &config).await?;
    Ok(config)
}

#[tauri::command]
pub async fn algoframe_product_delete_profile(name: String) -> Result<(), Error> {
    let db = db()?;
    let name = validate_profile_name(&name)?;
    let key = format!("product_profile:{name}");

    db.execute(Statement::from_string(
        DatabaseBackend::Sqlite,
        format!(
            "DELETE FROM algoframe_setting WHERE key='{}'",
            sql_string(&key)
        ),
    ))
    .await
    .map_err(|error| product_error("Profile:Delete", error))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_name_validation_accepts_normal_name() {
        assert_eq!(
            validate_profile_name("Balanced Main").unwrap(),
            "Balanced Main"
        );
    }

    #[test]
    fn profile_name_validation_rejects_empty_name() {
        assert!(validate_profile_name("   ").is_err());
    }
}
