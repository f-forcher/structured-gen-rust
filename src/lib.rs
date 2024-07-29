use regex_automata::{meta::Regex, Anchored, Input};

#[derive(Default, Debug)]
pub struct Mask {
    /// For simplicity we just use `u8` instead of
    /// a proper bitmask.
    /// A value of 1 means allowed and 0 forbidden.
    pub inner: Vec<u8>,
}

impl Mask {
    pub fn new(size: usize) -> Self {
        Mask {
            inner: vec![1; size],
        }
    }
}

/// Inefficient impl
pub fn naive_mask_from_pattern(
    vocabulary: &[String],
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
            println!("match ");
            println!("{}", pattern);
            println!("{}", possible_completion);
            mask.inner[i] = 1;
        } else {
            println!("No match ");
            mask.inner[i] = 0;
        }
    }

    mask
}

pub mod utils {
    use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};

    use crate::{naive_mask_from_pattern, Mask};

    // /// For simplicity, return this token if no other tokens are allowed.
    static DEFAULT_TOKEN: &str = "<NOTOK>";

    /// Configuration to select the token masking algorithm
    /// to select the valid next tokens.
    pub enum MaskingAlgo {
        /// Use naive O(N) pattern matching algorithm, i.e. for every
        /// token in the vocabulary check if the whole output would still
        /// validate the pattern. This requires O(N) steps where N
        /// is the total output sequence length.
        Naive,

        /// The algorithm from arxiv.org/abs/2307.09702, precomputing the
        /// token vocabulary with a hashmap from the pattern FSM states
        /// to valid tokens. The masking step is now O(1), indepentent
        /// of the current output sequence length.
        MapFSM, // TODO add the precomputed state here ie MapFSM(HashMap...)
    }

    pub trait LangModel {
        fn sample_one_token(&mut self, mask: Mask) -> &str;

        fn sample_n_tokens(
            &mut self,
            num_tokens: usize,
            previous_samples: &mut String,
            pattern: Option<&String>,
        ) -> String {
            for _i in 0..num_tokens {
                let mut mask = Mask::new(self.vocabulary_size());
                if let Some(pattern) = pattern {
                    mask =
                        naive_mask_from_pattern(self.get_vocabulary(), previous_samples, pattern);
                }

                let next_token: String = self.sample_one_token(mask).to_owned();
                previous_samples.push_str(&next_token);
            }

            previous_samples.to_string()
        }

        fn get_vocabulary(&self) -> &Vec<String>;
        fn vocabulary_size(&self) -> usize {
            self.get_vocabulary().len()
        }
    }

    pub struct ConstsLogitsModel<R: Rng> {
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

        fn get_vocabulary(&self) -> &Vec<String> {
            &self.vocabulary
        }
    }

    /// Simple mock language model that just
    /// iterates cyclically over its vocabulary.
    pub struct DeterministicModel {
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

        fn get_vocabulary(&self) -> &Vec<String> {
            &self.vocabulary
        }
    }
}
