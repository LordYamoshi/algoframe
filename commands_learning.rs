use serde_json::Value;
use utils::{get_location, Error};

#[tauri::command]
pub async fn learning_get_inspector() -> Result<Value, Error> {
    let path = std::path::PathBuf::from(utils::get_base_path())
        .join("algoframe_learning_inspector.json");

    if !path.exists() {
        return Ok(serde_json::json!({
            "model": "ml_bandit_v3",
            "status": "no_learning_data_yet",
            "message": "Run the live scraper and complete trades to create learning data."
        }));
    }

    let content = std::fs::read_to_string(&path).map_err(|error| {
        Error::new(
            "AlgoFrame:Learning:GetInspector",
            error.to_string(),
            get_location!(),
        )
    })?;

    serde_json::from_str(&content).map_err(|error| {
        Error::new(
            "AlgoFrame:Learning:ParseInspector",
            error.to_string(),
            get_location!(),
        )
    })
}

#[tauri::command]
pub async fn learning_get_dataset_path() -> Result<String, Error> {
    Ok(
        std::path::PathBuf::from(utils::get_base_path())
            .join("algoframe_learning_decisions.json")
            .to_string_lossy()
            .to_string(),
    )
}
