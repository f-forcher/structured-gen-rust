use rand::{
    distributions::{WeightedError, WeightedIndex},
    prelude::Distribution,
    Rng,
};
use regex_automata::{
    dfa::{dense, Automaton},
    util::primitives::StateID,
};

use crate::{naive_mask_from_pattern, MapAutomataStates, Mask, Token};

// /// Return this end-of-stream token if no other choices are allowed.
static EOS_TOKEN: &str = "<EOS>";

/// Configuration to select the token masking algorithm
/// to select the valid next tokens.
pub enum MaskingAlgo<'a> {
    /// Use naive O(N) pattern matching algorithm, i.e. for every
    /// token in the vocabulary check if the whole output would still
    /// validate the pattern. This requires O(N) steps where N
    /// is the total output sequence length.
    Naive { pattern: Option<&'a String> },

    /// The algorithm from arxiv.org/abs/2307.09702, precomputing the
    /// token vocabulary with a hashmap from the pattern FSM states
    /// to valid tokens. The masking step is now O(1), indepentent
    /// of the current output sequence length.
    CacheAutomataStates {
        map: MapAutomataStates,
        fsm: &'a dense::DFA<Vec<u32>>,
        current_state: &'a mut StateID,
    }, // TODO add the precomputed state here ie MapFSM(HashMap...)
}

pub trait LangModel {
    fn sample_one_token(&mut self, mask: Mask) -> &str;

    fn get_vocabulary(&self) -> &Vec<Token>;

    fn sample_n_tokens(
        &mut self,
        num_tokens: usize,
        previous_samples: &mut String,
        masking_algo: MaskingAlgo,
    ) -> String {
        match masking_algo {
            MaskingAlgo::Naive { ref pattern } => {
                for _i in 0..num_tokens {
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
                    previous_samples.push_str(&next_token);
                    // if next_token == EOS_TOKEN {
                    //     break;
                    // } else {
                    //     previous_samples.push_str(&next_token);
                    // }
                }
            }
            MaskingAlgo::CacheAutomataStates {
                ref map,
                fsm,
                current_state,
            } => {
                println!("Map {:?}", map);
                println!("State {:?}", current_state);
                for _i in 0..num_tokens {
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

                    // if i == num_tokens - 1 {
                    //     // This is the last token, must walk end of
                    //     *current_state = fsm.next_eoi_state(*current_state);
                    // } else {
                    //     *current_state = fsm.start_state_forward(&Input::new(&next_token)).unwrap();
                    // }

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
        // TODO possibility of optimization using update_weights
        // if few removed tokens
        // TODO handle error
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
    pub vocabulary: Vec<Token>,
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
