/*!
    Utilities for testing structured generation without the overhead of running a "real" Large Language Model.
*/

use rand::{
    distributions::{WeightedError, WeightedIndex},
    prelude::Distribution,
    Rng,
};
use regex_automata::dfa::Automaton;

use crate::{naive_mask_from_pattern, Mask, MaskingAlgo, Token};

// /// Return this end-of-stream token if no other choices are allowed.
static EOS_TOKEN: &str = "<EOS>";

pub trait LangModel {
    fn sample_one_token(&mut self, mask: Mask) -> &str;

    fn get_vocabulary(&self) -> &Vec<Token>;

    fn sample_multiple_tokens(
        &mut self,
        max_tokens: usize,
        previous_samples: &mut String,
        masking_algo: MaskingAlgo,
    ) -> String {
        match masking_algo {
            MaskingAlgo::Naive { ref pattern } => {
                for _i in 0..max_tokens {
                    let mut mask = Mask::ones(self.vocabulary_size());
                    if let Some(pattern) = pattern {
                        mask = naive_mask_from_pattern(
                            self.get_vocabulary(),
                            previous_samples,
                            pattern,
                        );
                    }
                    println!("naive mask: {:?}", mask);
                    let next_token: String = self.sample_one_token(mask).to_owned();

                    if next_token == EOS_TOKEN {
                        break;
                    } else {
                        previous_samples.push_str(&next_token);
                    }
                }
            }
            MaskingAlgo::CacheAutomataStates {
                ref map,
                fsm,
                current_state,
            } => {
                println!("Map {:?}", map);
                println!("State {:?}", current_state);
                for _i in 0..max_tokens {
                    let mut mask = Mask::zeros(self.vocabulary_size());

                    let vocab = self.get_vocabulary();

                    for (idx, bit) in mask.inner.iter_mut().enumerate() {
                        if map
                            .get(current_state)
                            .expect("key should exists for all states")
                            .contains(&vocab[idx])
                        {
                            *bit = 1;
                            println!("state {:?} contains {}", current_state, &vocab[idx]);
                        } else {
                            *bit = 0;
                            println!("state {:?} does not contain {}", current_state, &vocab[idx]);
                        }
                    }
                    println!();
                    println!("Mask {:?}", mask);

                    let next_token: String = self.sample_one_token(mask).to_owned();

                    if next_token == EOS_TOKEN {
                        break;
                    } else {
                        previous_samples.push_str(&next_token);
                    }

                    // Advance dfa
                    for &b in next_token.as_bytes().iter() {
                        *current_state = fsm.next_state(*current_state, b);
                    }
                }
            }
        }

        previous_samples.to_string()
    }

    fn vocabulary_size(&self) -> usize {
        self.get_vocabulary().len()
    }
}

pub struct ConstsLogitsModel<R: Rng> {
    pub vocabulary: Vec<Token>,
    pub dist: WeightedIndex<f64>,
    pub weights: Vec<f64>,
    rng: R,
}

impl<R: Rng> ConstsLogitsModel<R> {
    pub fn new_with_weights(vocabulary: Vec<Token>, weights: &[f64], rng: R) -> Self {
        ConstsLogitsModel {
            vocabulary,
            dist: WeightedIndex::new(weights).unwrap(),
            weights: weights.into(),
            rng,
        }
    }

    pub fn new(vocabulary: Vec<Token>, rng: R) -> Self {
        let weights = vec![1.; vocabulary.len()];
        ConstsLogitsModel::new_with_weights(vocabulary, &weights, rng)
    }
}

impl<R: Rng> LangModel for ConstsLogitsModel<R> {
    fn sample_one_token(&mut self, mask: Mask) -> &str {
        let new_weights: Vec<_> = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, w)| (mask.inner[i] as f64) * w) // Apply masking
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
            } else {
                continue;
            }
        }

        out_token
    }

    fn get_vocabulary(&self) -> &Vec<Token> {
        &self.vocabulary
    }
}
