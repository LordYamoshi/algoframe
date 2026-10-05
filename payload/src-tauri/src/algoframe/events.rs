
use std::sync::{Mutex, OnceLock};

use chrono::{DateTime, Duration, Utc};
use serde_json::Value;
use service::sea_orm::DatabaseConnection;
use utils::{get_location, Error};

use crate::HTTP_CLIENT;

use super::{
    store::LearningStore,
    types::EventSignal,
};

const EVENT_REFRESH_MINUTES: i64 = 30;
const WORLDSTATE_BASE: &str = "https://api.warframestat.us/pc";

static LAST_REFRESH: OnceLock<Mutex<Option<DateTime<Utc>>>> = OnceLock::new();

pub async fn refresh_event_intelligence(
    db: &DatabaseConnection,
) -> Result<Vec<EventSignal>, Error> {
    let lock = LAST_REFRESH.get_or_init(|| Mutex::new(None));

    let should_refresh = {
        let guard = lock.lock().map_err(|_| {
            Error::new(
                "AlgoFrame:Events:RefreshLock",
                "Event refresh lock is poisoned",
                get_location!(),
            )
        })?;

        guard
            .as_ref()
            .map(|last| Utc::now() - *last >= Duration::minutes(EVENT_REFRESH_MINUTES))
            .unwrap_or(true)
    };

    if !should_refresh {
        return LearningStore::load_active_events(db).await;
    }

    let client = HTTP_CLIENT.get_or_init(reqwest::Client::new);

    let endpoints = [
        ("events", format!("{WORLDSTATE_BASE}/events")),
        ("news", format!("{WORLDSTATE_BASE}/news")),
        ("void_trader", format!("{WORLDSTATE_BASE}/voidTrader")),
        ("vault_trader", format!("{WORLDSTATE_BASE}/vaultTrader")),
    ];

    let mut signals = Vec::new();

    for (kind, url) in endpoints {
        let response = match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => response,
            _ => continue,
        };

        let payload = match response.json::<Value>().await {
            Ok(payload) => payload,
            Err(_) => continue,
        };

        signals.extend(parse_worldstate(kind, &payload));
    }

    deduplicate_signals(&mut signals);

    for signal in &signals {
        LearningStore::upsert_event(db, signal).await?;
    }

    if let Ok(mut guard) = lock.lock() {
        *guard = Some(Utc::now());
    }

    LearningStore::load_active_events(db).await
}

fn parse_worldstate(kind: &str, payload: &Value) -> Vec<EventSignal> {
    match kind {
        "void_trader" => parse_trader(payload, "baro_kiteer", -0.45),
        "vault_trader" => parse_trader(payload, "prime_resurgence", -0.50),
        "events" => parse_generic_collection(payload, "world_event"),
        "news" => parse_generic_collection(payload, "news"),
        _ => vec![],
    }
}

fn parse_trader(
    payload: &Value,
    kind: &str,
    impact: f64,
) -> Vec<EventSignal> {
    let active = payload
        .get("active")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if !active {
        return vec![];
    }

    let now = Utc::now();
    let starts_at = parse_time(
        payload
            .get("activation")
            .or_else(|| payload.get("startString"))
            .or_else(|| payload.get("date")),
    )
    .unwrap_or(now);

    let ends_at = parse_time(
        payload
            .get("expiry")
            .or_else(|| payload.get("endString")),
    )
    .or_else(|| Some(now + Duration::days(2)));

    let trader_name = payload
        .get("character")
        .and_then(Value::as_str)
        .unwrap_or(kind);

    let inventory = payload
        .get("inventory")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    inventory
        .into_iter()
        .filter_map(|entry| {
            let title = entry
                .get("item")
                .or_else(|| entry.get("name"))
                .and_then(Value::as_str)?
                .trim()
                .to_string();

            if title.is_empty() {
                return None;
            }

            Some(EventSignal {
                id: format!(
                    "{}:{}:{}",
                    kind,
                    stable_id(&title),
                    starts_at.timestamp()
                ),
                kind: kind.to_string(),
                title: format!("{trader_name}: {title}"),
                keywords: keyword_tokens(&title),
                tags: vec![
                    if kind == "prime_resurgence" {
                        "prime".to_string()
                    } else {
                        "baro".to_string()
                    },
                ],
                impact,
                confidence: 0.90,
                starts_at,
                ends_at,
                source: "warframestat_worldstate".to_string(),
            })
        })
        .collect()
}

fn parse_generic_collection(
    payload: &Value,
    kind: &str,
) -> Vec<EventSignal> {
    let values = match payload {
        Value::Array(values) => values.clone(),
        Value::Object(_) => vec![payload.clone()],
        _ => vec![],
    };

    values
        .into_iter()
        .filter_map(|entry| {
            let title = text_field(
                &entry,
                &[
                    "description",
                    "message",
                    "title",
                    "tooltip",
                    "node",
                    "name",
                ],
            )?;

            let lower = title.to_lowercase();

            let (impact, confidence) = event_impact(&lower);

            // Generic worldstate noise should not influence market predictions
            // unless it contains a market-relevant keyword.
            if impact.abs() < 0.01 {
                return None;
            }

            let now = Utc::now();

            let starts_at = parse_time(
                entry
                    .get("activation")
                    .or_else(|| entry.get("date"))
                    .or_else(|| entry.get("start")),
            )
            .unwrap_or(now);

            let ends_at = parse_time(
                entry
                    .get("expiry")
                    .or_else(|| entry.get("end"))
                    .or_else(|| entry.get("endDate")),
            )
            .or_else(|| {
                Some(
                    starts_at
                        + if kind == "news" {
                            Duration::days(14)
                        } else {
                            Duration::days(7)
                        },
                )
            });

            if ends_at.map(|end| end < now).unwrap_or(false) {
                return None;
            }

            let mut keywords = keyword_tokens(&title);

            if lower.contains("prime resurgence") {
                keywords.push("prime".into());
                keywords.push("resurgence".into());
            }
            if lower.contains("prime access") {
                keywords.push("prime".into());
            }
            if lower.contains("vault") || lower.contains("unvault") {
                keywords.push("vault".into());
                keywords.push("prime".into());
            }

            keywords.sort();
            keywords.dedup();

            Some(EventSignal {
                id: format!(
                    "{}:{}:{}",
                    kind,
                    stable_id(&title),
                    starts_at.timestamp()
                ),
                kind: kind.to_string(),
                title,
                keywords,
                tags: vec![],
                impact,
                confidence,
                starts_at,
                ends_at,
                source: "warframestat_worldstate".to_string(),
            })
        })
        .collect()
}

fn event_impact(lower: &str) -> (f64, f64) {
    if lower.contains("prime resurgence")
        || lower.contains("unvault")
        || lower.contains("unvaulted")
        || lower.contains("returns")
    {
        return (-0.45, 0.75);
    }

    if lower.contains("vaulted")
        || lower.contains("entering the vault")
        || lower.contains("retiring")
    {
        return (0.45, 0.75);
    }

    if lower.contains("prime access")
        || lower.contains("new prime")
        || lower.contains("prime arrives")
    {
        return (-0.25, 0.55);
    }

    if lower.contains("baro")
        || lower.contains("void trader")
    {
        return (-0.20, 0.60);
    }

    if lower.contains("buff")
        || lower.contains("rework")
        || lower.contains("balance")
        || lower.contains("hotfix")
        || lower.contains("update")
    {
        // Direction is intentionally mild because patch-note semantics are not
        // known from a headline alone. The main effect is lowering reliance on
        // stale historical data for named items.
        return (0.08, 0.35);
    }

    (0.0, 0.0)
}

fn text_field(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(text) = value.get(*key).and_then(Value::as_str) {
            let text = strip_markup(text).trim().to_string();
            if !text.is_empty() {
                return Some(text);
            }
        }
    }

    None
}

fn parse_time(value: Option<&Value>) -> Option<DateTime<Utc>> {
    let value = value?;

    if let Some(text) = value.as_str() {
        if let Ok(parsed) = DateTime::parse_from_rfc3339(text) {
            return Some(parsed.with_timezone(&Utc));
        }

        if let Ok(timestamp) = text.parse::<i64>() {
            return DateTime::from_timestamp(timestamp, 0);
        }
    }

    if let Some(timestamp) = value.as_i64() {
        // Warframe-related APIs sometimes use milliseconds.
        let seconds = if timestamp > 10_000_000_000 {
            timestamp / 1000
        } else {
            timestamp
        };

        return DateTime::from_timestamp(seconds, 0);
    }

    None
}

fn keyword_tokens(value: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "with", "from", "this", "that", "your", "have", "will", "into",
        "prime", "update", "hotfix", "warframe", "available", "returns",
        "official", "news", "event", "the", "and", "for", "you", "new",
    ];

    let mut result: Vec<String> = value
        .split(|character: char| !character.is_alphanumeric() && character != '\'')
        .map(|part| part.trim().to_lowercase())
        .filter(|part| part.len() >= 4)
        .filter(|part| !STOP.contains(&part.as_str()))
        .collect();

    // Also retain two-word phrases; item names often need more context than a
    // single token (e.g. "Xaku Prime", "Primed Flow").
    let words = result.clone();

    for pair in words.windows(2) {
        result.push(format!("{} {}", pair[0], pair[1]));
    }

    result.sort();
    result.dedup();
    result
}

fn strip_markup(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut in_tag = false;

    for character in value.chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(character),
            _ => {}
        }
    }

    result
}

fn stable_id(value: &str) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.to_lowercase().hash(&mut hasher);
    hasher.finish()
}

fn deduplicate_signals(signals: &mut Vec<EventSignal>) {
    signals.sort_by(|a, b| a.id.cmp(&b.id));
    signals.dedup_by(|a, b| a.id == b.id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resurgence_is_supply_negative_for_price() {
        let (impact, confidence) = event_impact("prime resurgence returns");
        assert!(impact < 0.0);
        assert!(confidence > 0.5);
    }

    #[test]
    fn vaulting_is_scarcity_positive() {
        let (impact, _) = event_impact("xaku prime entering the vault");
        assert!(impact > 0.0);
    }

    #[test]
    fn extracts_item_tokens() {
        let tokens = keyword_tokens("Xaku Prime Systems");
        assert!(tokens.iter().any(|token| token == "xaku"));
        assert!(tokens.iter().any(|token| token.contains("systems")));
    }
}
