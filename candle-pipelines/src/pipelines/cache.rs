use crate::error::Result;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

pub trait ModelOptions {
    fn cache_key(&self) -> String;
}

// Cache stores WEAK references - models are freed when all pipelines using them drop.
// CudaDevice caching (in utils/mod.rs) ensures reloads use the same stream.
type CacheStorage = HashMap<(TypeId, String), Box<dyn Any + Send + Sync>>;

pub struct ModelCache {
    cache: Arc<Mutex<CacheStorage>>,
}

impl ModelCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn get_or_create<M, F>(&self, key: &str, loader: F) -> Result<Arc<M>>
    where
        M: Send + Sync + 'static,
        F: FnOnce() -> Result<M>,
    {
        let type_id = TypeId::of::<M>();
        let cache_key = (type_id, key.to_string());

        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(boxed) = cache.get(&cache_key) {
                // Try to upgrade weak ref
                if let Some(weak) = boxed.downcast_ref::<Weak<M>>() {
                    if let Some(strong) = weak.upgrade() {
                        return Ok(strong);
                    }
                }
                // Weak ref dead, remove stale entry
                cache.remove(&cache_key);
            }
        }

        let model = Arc::new(loader()?);

        {
            let mut cache = self.cache.lock().unwrap();
            let weak: Weak<M> = Arc::downgrade(&model);
            cache.insert(cache_key, Box::new(weak));
        }

        Ok(model)
    }

    #[allow(dead_code)]
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        let cache = self.cache.lock().unwrap();
        cache.is_empty()
    }

    pub async fn get_or_create_async<M, Fut, F>(&self, key: &str, loader: F) -> Result<Arc<M>>
    where
        M: Send + Sync + 'static,
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<M>>,
    {
        let type_id = TypeId::of::<M>();
        let cache_key = (type_id, key.to_string());

        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(boxed) = cache.get(&cache_key) {
                if let Some(weak) = boxed.downcast_ref::<Weak<M>>() {
                    if let Some(strong) = weak.upgrade() {
                        return Ok(strong);
                    }
                }
                cache.remove(&cache_key);
            }
        }

        let model = Arc::new(loader().await?);

        {
            let mut cache = self.cache.lock().unwrap();
            let weak: Weak<M> = Arc::downgrade(&model);
            cache.insert(cache_key, Box::new(weak));
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

    #[test]
    fn test_different_keys_independent() {
        let cache = ModelCache::new();

        let model1 = cache
            .get_or_create::<TestModel, _>("key1", || Ok(TestModel { id: "first".into() }))
            .unwrap();

        let model2 = cache
            .get_or_create::<TestModel, _>("key2", || {
                Ok(TestModel {
                    id: "second".into(),
                })
            })
            .unwrap();

        assert_eq!(model1.id, "first");
        assert_eq!(model2.id, "second");
        assert_eq!(cache.len(), 2);
    }
}
