/*!
    Utilities for testing structured generation without the overhead of running a "real" Large Language Model.
*/

use log::debug;
use rand::{
    distributions::{WeightedError, WeightedIndex},
    prelude::Distribution,
    Rng,
};

use crate::{LangModel, Mask, Token, EOS_TOKEN};

pub struct ConstLogitsModel<R: Rng> {
    pub vocabulary: Vec<Token>,
    pub dist: WeightedIndex<f64>,
    pub weights: Vec<f64>,
    rng: R,
}

impl<R: Rng> ConstLogitsModel<R> {
    pub fn new_with_weights(vocabulary: Vec<Token>, weights: &[f64], rng: R) -> Self {
        ConstLogitsModel {
            vocabulary,
            dist: WeightedIndex::new(weights).unwrap(),
            weights: weights.into(),
            rng,
        }
    }

    pub fn new(vocabulary: Vec<Token>, rng: R) -> Self {
        let weights = vec![1.; vocabulary.len()];
        ConstLogitsModel::new_with_weights(vocabulary, &weights, rng)
    }
}

impl<R: Rng> LangModel for ConstLogitsModel<R> {
    fn sample_one_token(&mut self, mask: Mask) -> &str {
        let new_weights: Vec<_> = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, w)| f64::from(mask.inner[i]) * w) // Apply masking
            .collect();

        self.dist = match WeightedIndex::new(new_weights) {
            Ok(weighted_index) => weighted_index,
            // If no token is allowed, return end-of-stream token.
            Err(WeightedError::AllWeightsZero) => return EOS_TOKEN,
            _ => todo!("error handling"),
        };
        &self.vocabulary[self.dist.sample(&mut self.rng)]
    }

    fn get_vocabulary(&self) -> &Vec<Token> {
        &self.vocabulary
    }
}

/// Simple mock language model that just
/// iterates cyclically over its vocabulary.
pub struct DeterministicModel {
    vocabulary: Vec<Token>,
    idx: usize,
}

impl DeterministicModel {
    pub fn new(vocabulary: Vec<Token>) -> Self {
        DeterministicModel { vocabulary, idx: 0 }
    }
}

impl LangModel for DeterministicModel {
    fn sample_one_token(&mut self, mask: Mask) -> &str {
        let mut out_token = EOS_TOKEN;

        for i in 0..self.vocabulary_size() {
            let cyclic_idx = (self.idx + i) % self.vocabulary_size();
            if mask.inner[cyclic_idx] != 0 {
                out_token = &self.vocabulary[cyclic_idx];
                self.idx = (self.idx + 1) % self.vocabulary_size();
                break;
            }
        }

        debug!(
            "Next tokens allowed: {:?}",
            mask.inner
                .iter()
                .enumerate()
                .filter(|&(_i, m)| (*m != 0))
                .map(|(i, _m)| self.get_vocabulary()[i].clone())
                .collect::<Vec<_>>()
        );

        out_token
    }

    fn get_vocabulary(&self) -> &Vec<Token> {
        &self.vocabulary
    }
}
