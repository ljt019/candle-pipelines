//! Model caching utilities for sharing weights across multiple pipelines.

use crate::Result;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Trait implemented by model option types to generate a stable cache key.
pub trait ModelOptions {
    fn cache_key(&self) -> String;
}

type CacheStorage = HashMap<(TypeId, String), Arc<dyn Any + Send + Sync>>;

/// A thread-safe cache for model instances.
pub struct ModelCache {
    cache: Arc<Mutex<CacheStorage>>,
}

impl ModelCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn get_or_create<M, F>(&self, key: &str, loader: F) -> Result<M>
    where
        M: Clone + Send + Sync + 'static,
        F: FnOnce() -> Result<M>,
    {
        let type_id = TypeId::of::<M>();
        let cache_key = (type_id, key.to_string());

        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                if let Some(model) = cached.downcast_ref::<M>() {
                    return Ok(model.clone());
                }
            }
        }

        let model = loader()?;

        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(
                cache_key,
                Arc::new(model.clone()) as Arc<dyn Any + Send + Sync>,
            );
        }

        Ok(model)
    }

    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    pub fn len(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.len()
    }

    pub fn is_empty(&self) -> bool {
        let cache = self.cache.lock().unwrap();
        cache.is_empty()
    }

    /// Async version for models with async constructors (e.g., TextGenerationModel).
    pub async fn get_or_create_async<M, Fut, F>(&self, key: &str, loader: F) -> Result<M>
    where
        M: Clone + Send + Sync + 'static,
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<M>>,
    {
        let type_id = TypeId::of::<M>();
        let cache_key = (type_id, key.to_string());

        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                if let Some(model) = cached.downcast_ref::<M>() {
                    return Ok(model.clone());
                }
            }
        }

        let model = loader().await?;

        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(
                cache_key,
                Arc::new(model.clone()) as Arc<dyn Any + Send + Sync>,
            );
        }

        Ok(model)
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

static GLOBAL_MODEL_CACHE: once_cell::sync::Lazy<ModelCache> =
    once_cell::sync::Lazy::new(ModelCache::new);

pub fn global_cache() -> &'static ModelCache {
    &GLOBAL_MODEL_CACHE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestModel {
        id: String,
    }

    #[test]
    fn test_cache_returns_same_instance() {
        let cache = ModelCache::new();
        let model1 = cache
            .get_or_create::<TestModel, _>("test", || {
                Ok(TestModel {
                    id: "original".into(),
                })
            })
            .unwrap();
        let model2 = cache
            .get_or_create::<TestModel, _>("test", || Ok(TestModel { id: "new".into() }))
            .unwrap();
        assert_eq!(model1.id, model2.id);
    }

    #[test]
    fn test_cache_clear() {
        let cache = ModelCache::new();
        #[derive(Clone)]
        struct A;
        let _ = cache.get_or_create::<A, _>("k", || Ok(A)).unwrap();
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
    }
}
