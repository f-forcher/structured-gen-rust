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

/// Model that randomly samples from a weighted list of tokens.
pub struct RandomSampleModel<R: Rng> {
    pub vocabulary: Vec<Token>,
    pub dist: WeightedIndex<f64>,
    pub weights: Vec<f64>,
    rng: R,
}

impl<R: Rng> RandomSampleModel<R> {
    pub fn new_with_weights(vocabulary: &[Token], weights: &[f64], rng: R) -> Self {
        RandomSampleModel {
            vocabulary: vocabulary.to_vec(),
            dist: WeightedIndex::new(weights).unwrap(),
            weights: weights.into(),
            rng,
        }
    }

    pub fn new(vocabulary: &[Token], rng: R) -> Self {
        let weights = vec![1.; vocabulary.len()];
        RandomSampleModel::new_with_weights(vocabulary, &weights, rng)
    }
}

impl<R: Rng> LangModel for RandomSampleModel<R> {
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

        debug!(
            "Next tokens allowed: {:?}",
            mask.inner
                .iter()
                .enumerate()
                .filter(|&(_i, m)| (*m != 0))
                .map(|(i, _m)| self.get_vocabulary()[i].clone())
                .collect::<Vec<_>>()
        );

        &self.vocabulary[self.dist.sample(&mut self.rng)]
    }

    fn get_vocabulary(&self) -> &[Token] {
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
    pub fn new(vocabulary: &[Token]) -> Self {
        DeterministicModel {
            vocabulary: vocabulary.to_vec(),
            idx: 0,
        }
    }
}

impl LangModel for DeterministicModel {
    fn sample_one_token(&mut self, mask: Mask) -> &str {
        let mut out_token = EOS_TOKEN;

        for i in 1..self.vocabulary_size() {
            let cyclic_idx = (self.idx + i) % self.vocabulary_size();
            if mask.inner[cyclic_idx] != 0 {
                out_token = &self.vocabulary[cyclic_idx];
                self.idx = cyclic_idx % self.vocabulary_size();
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

    fn get_vocabulary(&self) -> &[Token] {
        &self.vocabulary
    }
}

/// Helper function to generate a dictionary with up to num_tokens,
/// for testing and benchmarking. The dictionary consists of the
/// single chars a-z A-Z 0-9 . : , ! ? and every multiple char
/// combination of these, up to `num_tokens`.
pub fn generate_dict(num_tokens: usize) -> Vec<Token> {
    // Create a vector to hold the characters
    let mut chars: Vec<String> = Vec::new();

    for c in 'a'..='z' {
        chars.push(c.to_string());
    }
    for c in 'A'..='Z' {
        chars.push(c.to_string());
    }
    for c in '0'..='9' {
        chars.push(c.to_string());
    }
    let punctuation = ['.', ';', ',', '!', '?'];
    for &c in &punctuation {
        chars.push(c.to_string());
    }

    chars
        .clone()
        .into_iter()
        .chain(chars.iter().flat_map(|prefix| {
            chars.iter().map(|suffix| {
                let mut new_string = prefix.clone();
                new_string.push_str(suffix);
                new_string
            })
        }))
        .take(num_tokens)
        .collect()
}
