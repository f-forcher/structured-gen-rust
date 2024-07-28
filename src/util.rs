use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use structured_gen_rust::Mask;

// /// For simplicity, return this token if no other tokens are allowed.
static DEFAULT_TOKEN: &str = "<NOTOK>";

trait LangModel {
    fn sample_one_token(&mut self, mask: Mask) -> &str;

    fn sample_n_tokens(&mut self, num_tokens: usize) -> String {
        (0..num_tokens)
            .map(|_| {
                // TODO Contruct mask here
                let mask = Mask::new(self.vocabulary_size());
                self.sample_one_token(mask).to_owned()
            })
            .collect()
    }

    fn vocabulary_size(&self) -> usize;
}

struct ConstsLogitsModel<R: Rng> {
    pub vocabulary: Vec<String>,
    pub dist: WeightedIndex<f64>,
    pub weights: Vec<f64>,
    rng: R,
}

impl<R: Rng> ConstsLogitsModel<R> {
    pub fn new(vocabulary: Vec<String>, weights: &[f64], rng: R) -> Self {
        ConstsLogitsModel {
            vocabulary,
            dist: WeightedIndex::new(weights).unwrap(),
            weights: weights.into(),
            rng,
        }
    }
}

impl<R: Rng> LangModel for ConstsLogitsModel<R> {
    fn sample_one_token(&mut self, mask: Mask) -> &str {
        let new_weights: Vec<_> = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, w)| (mask.inner[i] as f64) * w)
            .collect();
        // TODO possibility of optimization using update_weights
        // if few removed tokens
        // TODO handle error
        self.dist = WeightedIndex::new(new_weights).unwrap();
        &self.vocabulary[self.dist.sample(&mut self.rng)]
    }
    
    fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Simple mock language model that just
/// iterates cyclically over its vocabulary.
struct DeterministicModel {
    pub vocabulary: Vec<String>,
    idx: usize,
}

impl DeterministicModel {
    pub fn new(vocabulary: Vec<String>) -> Self {
        DeterministicModel { vocabulary, idx: 0 }
    }
}

impl LangModel for DeterministicModel {
    fn sample_one_token(&mut self, mask: Mask) -> &str {
        let mut out_token = DEFAULT_TOKEN;

        for i in (0..self.vocabulary_size()) {
            let cyclic_idx = (self.idx + i) % &self.vocabulary_size();
            if mask.inner[cyclic_idx] != 0 {
                out_token = &self.vocabulary[cyclic_idx];
                break
            } else {
                continue
            }
        }
        out_token
    }

    fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}
