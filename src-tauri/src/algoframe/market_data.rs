use std::sync::{Mutex, OnceLock};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utils::{get_location, Error};

use crate::HTTP_CLIENT;

const RELIC_REFRESH_HOURS: i64 = 6;
const RELIC_DROP_URL: &str = "https://drops.warframestat.us/data/relics.json";

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RelicDropReward {
    pub item_name: String,
    pub chance: f64,
    pub rarity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RelicDropTable {
    pub tier: String,
    pub relic_name: String,
    pub state: String,
    pub rewards: Vec<RelicDropReward>,
}

#[derive(Clone, Debug, Default)]
struct CachedRelics {
    refreshed_at: Option<DateTime<Utc>>,
    tables: Vec<RelicDropTable>,
}

static RELIC_CACHE: OnceLock<Mutex<CachedRelics>> = OnceLock::new();

pub async fn relic_drop_tables() -> Result<Vec<RelicDropTable>, Error> {
    let cache = RELIC_CACHE.get_or_init(|| Mutex::new(CachedRelics::default()));

    if let Ok(guard) = cache.lock() {
        if guard
            .refreshed_at
            .map(|at| Utc::now() - at < Duration::hours(RELIC_REFRESH_HOURS))
            .unwrap_or(false)
            && !guard.tables.is_empty()
        {
            return Ok(guard.tables.clone());
        }
    }

    let client = HTTP_CLIENT.get_or_init(reqwest::Client::new);

    let response = client.get(RELIC_DROP_URL).send().await.map_err(|error| {
        Error::new(
            "AlgoFrame:MarketData:Relics:Request",
            error.to_string(),
            get_location!(),
        )
    })?;

    if !response.status().is_success() {
        return Err(Error::new(
            "AlgoFrame:MarketData:Relics:Status",
            format!("Relic drop endpoint returned {}", response.status()),
            get_location!(),
        ));
    }

    let payload = response.json::<Value>().await.map_err(|error| {
        Error::new(
            "AlgoFrame:MarketData:Relics:Json",
            error.to_string(),
            get_location!(),
        )
    })?;

    let tables = parse_relic_tables(&payload);

    if tables.is_empty() {
        return Err(Error::new(
            "AlgoFrame:MarketData:Relics:Empty",
            "Relic drop endpoint returned no parseable relic tables",
            get_location!(),
        ));
    }

    if let Ok(mut guard) = cache.lock() {
        guard.refreshed_at = Some(Utc::now());
        guard.tables = tables.clone();
    }

    Ok(tables)
}

fn parse_relic_tables(payload: &Value) -> Vec<RelicDropTable> {
    let rows = payload
        .get("relics")
        .and_then(Value::as_array)
        .or_else(|| payload.as_array());

    let Some(rows) = rows else {
        return vec![];
    };

    let mut result = Vec::new();

    for row in rows {
        let tier = row
            .get("tier")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();

        let relic_name = row
            .get("relicName")
            .or_else(|| row.get("relic_name"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();

        if tier.is_empty() || relic_name.is_empty() {
            continue;
        }

        if let Some(rewards) = row.get("rewards").and_then(Value::as_array) {
            let state = row
                .get("state")
                .and_then(Value::as_str)
                .unwrap_or("Intact")
                .to_string();

            let rewards = parse_rewards(rewards);

            if !rewards.is_empty() {
                result.push(RelicDropTable {
                    tier: tier.clone(),
                    relic_name: relic_name.clone(),
                    state,
                    rewards,
                });
            }

            continue;
        }

        if let Some(states) = row.get("rewards").and_then(Value::as_object) {
            for (state, rewards) in states {
                let Some(rewards) = rewards.as_array() else {
                    continue;
                };

                let rewards = parse_rewards(rewards);

                if rewards.is_empty() {
                    continue;
                }

                result.push(RelicDropTable {
                    tier: tier.clone(),
                    relic_name: relic_name.clone(),
                    state: state.clone(),
                    rewards,
                });
            }
        }
    }

    result
}

fn parse_rewards(rows: &[Value]) -> Vec<RelicDropReward> {
    rows.iter()
        .filter_map(|row| {
            let item_name = row
                .get("itemName")
                .or_else(|| row.get("item_name"))
                .and_then(Value::as_str)?
                .trim()
                .to_string();

            let chance = row.get("chance").and_then(Value::as_f64).unwrap_or(0.0);

            if item_name.is_empty() || chance <= 0.0 {
                return None;
            }

            Some(RelicDropReward {
                item_name,
                chance,
                rarity: row
                    .get("rarity")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            })
        })
        .collect()
}

pub fn parse_relic_identity(item_name: &str) -> Option<(String, String)> {
    let clean = item_name.replace(" Relic", "").replace(" relic", "");

    let mut parts = clean.split_whitespace();
    let tier = parts.next()?.trim().to_string();
    let relic_name = parts.next()?.trim().to_string();

    if !matches!(
        tier.to_lowercase().as_str(),
        "lith" | "meso" | "neo" | "axi" | "requiem"
    ) {
        return None;
    }

    Some((tier, relic_name))
}

pub fn normalized_refinement(value: Option<&str>) -> String {
    match value.unwrap_or("Intact").to_lowercase().as_str() {
        "exceptional" => "Exceptional".to_string(),
        "flawless" => "Flawless".to_string(),
        "radiant" => "Radiant".to_string(),
        _ => "Intact".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_relic_name() {
        assert_eq!(
            parse_relic_identity("Axi A1 Relic"),
            Some(("Axi".to_string(), "A1".to_string()))
        );
    }

    #[test]
    fn refinement_defaults_to_intact() {
        assert_eq!(normalized_refinement(None), "Intact");
    }
}
