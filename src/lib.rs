use anyhow::{anyhow, Result};
use regex_automata::{
    dfa::{self, dense, Automaton},
    MatchError,
};
use std::collections::HashMap;

use regex_automata::{meta::Regex, util::primitives::StateID, Anchored, Input};

#[derive(Default, Debug)]
pub struct Mask {
    /// For simplicity we just use `u8` instead of
    /// a proper bitmask.
    /// A value of 1 means allowed and 0 forbidden.
    pub inner: Vec<u8>,
}

type Token = String;
type MapAutomataStates = HashMap<StateID, Vec<Token>>;

impl Mask {
    pub fn new(size: usize) -> Self {
        Mask {
            inner: vec![1; size],
        }
    }
}

/// Inefficient impl
pub fn naive_mask_from_pattern(
    vocabulary: &[Token],
    previous_samples: &str,
    pattern: &str,
) -> Mask {
    // TODO save the regex in struct?
    let mut mask = Mask::new(vocabulary.len());

    for (i, tok) in vocabulary.iter().enumerate() {
        let mut possible_completion = previous_samples.to_owned();
        possible_completion.push_str(tok);

        let possible_completion2 = Input::new(&possible_completion).anchored(Anchored::Yes);

        let re = Regex::new(pattern).unwrap();
        if re.is_match(possible_completion2) {
            mask.inner[i] = 1;
        } else {
            mask.inner[i] = 0;
        }
    }

    mask
}

pub fn find_subsequences(fsm: &dense::DFA<Vec<u32>>, token: &Token) -> Result<Vec<Vec<StateID>>> {
    let mut subsequences = vec![];
    // let classes = fsm.byte_classes();

    'states_loop: for state in fsm.tt.states() {
        //let start_state = fsm.start_state_forward(&Input::new(token))?;

        let mut state_sequences = vec![];
        let mut curr_state = state.id();

        for &b in token.as_bytes().iter() {
            if fsm.is_special_state(curr_state) {
                if fsm.is_dead_state(curr_state) || fsm.is_quit_state(curr_state) {
                    state_sequences.clear(); // Needed?
                    continue 'states_loop;
                }
            }

            state_sequences.push(curr_state);

            curr_state = fsm.next_state(curr_state, b);
        }
        subsequences.push(state_sequences);
    }

    Ok(subsequences)
}

pub fn create_states_map(fsm: &dense::DFA<Vec<u32>>, vocabulary: &[Token]) -> MapAutomataStates {
    let mut map = MapAutomataStates::new();
    for token in vocabulary {
        let subsequences = find_subsequences(fsm, token).unwrap();
        for sequence in subsequences {
            // TODO do we need the rest of the sequence?
            map.entry(sequence[0])
                .and_modify(|tokens| tokens.push(token.clone()))
                .or_default();
        }
    }
    map
}

// /// Efficient impl using Automata States map
// pub fn mask_from_pattern_using_map(
//     vocabulary: &[Token],
//     previous_samples: &str,
//     pattern: &str,
//     map: MapAutomataStates,
// ) -> Mask {
//     // TODO save the regex in struct?
//     let mut mask = Mask::new(vocabulary.len());

//     for (i, tok) in vocabulary.iter().enumerate() {
//         let mut possible_completion = previous_samples.to_owned();
//         possible_completion.push_str(tok);

//         let possible_completion2 = Input::new(&possible_completion).anchored(Anchored::Yes);

//         let re = Regex::new(pattern).unwrap();
//         if re.is_match(possible_completion2) {
//             mask.inner[i] = 1;
//         } else {
//             mask.inner[i] = 0;
//         }
//     }

//     mask
// }

pub mod utils {
    use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
    use regex_automata::{dfa::dense, util::primitives::StateID};

    use crate::{naive_mask_from_pattern, MapAutomataStates, Mask, Token};

    // /// For simplicity, return this token if no other tokens are allowed.
    static DEFAULT_TOKEN: &str = "<NOTOK>";

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
            current_state: StateID,
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
            for _i in 0..num_tokens {
                match masking_algo {
                    MaskingAlgo::Naive { ref pattern } => {
                        let mut mask = Mask::new(self.vocabulary_size());
                        if let Some(pattern) = pattern {
                            mask = naive_mask_from_pattern(
                                self.get_vocabulary(),
                                previous_samples,
                                pattern,
                            );
                        }

                        let next_token: String = self.sample_one_token(mask).to_owned();
                        previous_samples.push_str(&next_token);
                    }
                    MaskingAlgo::CacheAutomataStates {
                        ref map,
                        fsm,
                        current_state,
                    } => {
                        todo!()
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
        pub fn new(vocabulary: Vec<Token>, weights: &[f64], rng: R) -> Self {
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
                .map(|(i, w)| (mask.inner[i] as f64) * w) // Apply masking
                .collect();
            // TODO possibility of optimization using update_weights
            // if few removed tokens
            // TODO handle error
            self.dist = WeightedIndex::new(new_weights).unwrap();
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
            let mut out_token = DEFAULT_TOKEN;

            for i in 0..self.vocabulary_size() {
                let cyclic_idx = (self.idx + i) % self.vocabulary_size();
                if mask.inner[cyclic_idx] != 0 {
                    out_token = &self.vocabulary[cyclic_idx];
                    self.idx += 1;
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
}
