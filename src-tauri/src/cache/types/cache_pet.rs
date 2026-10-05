use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct CachePet {
    #[serde(flatten)]
    pub base: super::cache_item_base::CacheItemBase,
}
