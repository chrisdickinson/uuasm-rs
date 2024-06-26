use std::{collections::HashMap, sync::Arc};

#[derive(Default, Debug, Clone)]
pub(crate) struct InternMap {
    strings: Vec<Arc<str>>,
    string_to_idx: HashMap<Arc<str>, usize>,
}

impl InternMap {
    pub(crate) fn get(&self, key: &str) -> Option<usize> {
        self.string_to_idx.get(key).copied()
    }

    pub(crate) fn idx(&self, key: usize) -> Option<&str> {
        self.strings.get(key).map(|xs| xs.as_ref())
    }

    pub(crate) fn insert(&mut self, string: &str) -> usize {
        if let Some(idx) = self.string_to_idx.get(string) {
            return *idx;
        }

        let string: Arc<str> = string.into();
        let idx = self.strings.len();
        self.strings.push(string.clone());
        self.string_to_idx.insert(string, idx);
        idx
    }
}


