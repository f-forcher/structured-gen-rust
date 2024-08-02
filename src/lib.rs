/*!
    Main lib module containing the routines to structurally generate text from LLMs.
*/

use anyhow::Result;
use log::{debug, info, trace};
use regex_automata::{
    dfa::{dense, Automaton},
    meta::Regex,
    util::primitives::StateID,
    Input,
};
use std::collections::{HashMap, HashSet};

pub mod util;

// /// Return this end-of-stream token if no other choices are allowed.
static EOS_TOKEN: &str = "<EOS>";

/// The bitmask that is used to set to zero the LLM logits
/// according to the given pattern.
#[derive(Default, Debug)]
pub struct Mask {
    /// For simplicity at the moment `u8` is used instead of
    /// just a bit.
    ///
    /// A value of `1` means the token at the corresponding position in the dictionary is allowed,
    ///  and `0` that the token is forbidden.
    pub inner: Vec<u8>,
}

/// Select and configure the token masking algorithm
/// to select the valid next tokens.
pub enum MaskingAlgorithmConfig<'a> {
    /// Do not perform structured generation, mask will allow all tokens
    NoMasking,

    /// Use naive O(N) pattern matching algorithm, i.e. for every
    /// token in the vocabulary check if the whole output would still
    /// validate the pattern. This requires O(N) steps where N
    /// is the total output sequence length.
    Naive(Pattern<'a>),

    /// The algorithm from arxiv.org/abs/2307.09702, precomputing the
    /// token vocabulary with a hashmap from the pattern FSM states
    /// to valid tokens. The masking step is now O(1), indepentent
    /// of the current output sequence length.
    IndexedFSM(Pattern<'a>),
}

pub struct StatefulFSM {
    pub inner: dense::DFA<Vec<u32>>,
    pub current_state: StateID,
}

/// The internal state for the different masking algorithms.
pub enum MaskingAlgoState<'a> {
    /// Do not perform structured generation, mask will allow all tokens.
    /// No state necessary.
    NoMasking,

    /// Use naive O(N) pattern matching algorithm, i.e. for every
    /// token in the vocabulary check if the whole output would still
    /// validate the pattern. This requires O(N) steps where N
    /// is the total output sequence length.
    Naive { pattern: Pattern<'a> },

    /// The algorithm from arxiv.org/abs/2307.09702, precomputing the
    /// token vocabulary with a hashmap from the pattern FSM states
    /// to valid tokens. The masking step is now O(1), indepentent
    /// of the current output sequence length.
    IndexedFSM {
        map: IndexedFSMStates,
        fsm: Box<StatefulFSM>,
    }, // TODO add the precomputed state here ie MapFSM(HashMap...)
}

impl<'a> MaskingAlgoState<'a> {
    fn from_config(
        config: &MaskingAlgorithmConfig<'a>,
        input_prompt: &'a str,
        vocabulary: &[Token],
    ) -> Self {
        match config {
            MaskingAlgorithmConfig::NoMasking => MaskingAlgoState::NoMasking,
            MaskingAlgorithmConfig::Naive(pattern) => MaskingAlgoState::Naive { pattern },
            MaskingAlgorithmConfig::IndexedFSM(pattern) => {
                let fsm = dense::DFA::new(pattern).expect("Valid regex pattern");
                let mut state = fsm
                    .start_state_forward(&Input::new(input_prompt))
                    .expect("todo err");

                // Advance fsm over previous input
                for &b in input_prompt.as_bytes() {
                    state = fsm.next_state(state, b);
                }

                let map = map_states_to_vocab(&fsm, vocabulary);
                MaskingAlgoState::IndexedFSM {
                    map,
                    fsm: Box::new(StatefulFSM {
                        inner: fsm,
                        current_state: state,
                    }),
                }
            }
        }
    }
}

// TODO use proper newtypes enforcing invariants (no empty string token etc).

/// A string representing a component of a Language Model alphabet.
type Token = String;

/// A string representing a component of a Language Model alphabet.
type Pattern<'a> = &'a str;

/// Index from the fsm states to the set of tokens accepted by that state.
type IndexedFSMStates = HashMap<StateID, HashSet<Token>>;

/// An ordered sequence of states of a given FSM, representing paths traversed by
/// the FSM when processing a given string.
type StateSequence = Vec<StateID>;

impl Mask {
    /// Return a mask of `1`s, every token is allowed.
    ///
    /// `size` should be the length of the corresponding dictionary.
    pub fn ones(size: usize) -> Self {
        Mask {
            inner: vec![1; size],
        }
    }

    /// Return a mask of `0`s, no token is allowed.
    ///
    /// `size` should be the length of the corresponding dictionary.
    pub fn zeros(size: usize) -> Self {
        Mask {
            inner: vec![0; size],
        }
    }
}

pub trait LangModel {
    fn sample_one_token(&mut self, mask: Mask) -> &str;

    fn get_vocabulary(&self) -> &Vec<Token>;

    // Sample up to `max_tokens` and append them to `text_buffer`.
    fn sample_multiple_tokens(
        &mut self,
        max_tokens: usize,
        text_buffer: &mut String,
        mask_algorithm: MaskingAlgoState,
    ) {
        match mask_algorithm {
            MaskingAlgoState::NoMasking => {
                info!("Using NoMasking algorithm");

                for _i in 0..max_tokens {
                    // Identity mask
                    let mask = Mask::ones(self.vocabulary_size());

                    let next_token: String = self.sample_one_token(mask).to_owned();

                    if next_token == EOS_TOKEN {
                        debug!("EOS token received, returning early from stream");
                        break;
                    } else {
                        text_buffer.push_str(&next_token);
                    }
                }
            }
            MaskingAlgoState::Naive { pattern } => {
                info!("Using Naive algorithm");

                for i in 0..max_tokens {
                    let mask = naive_mask_from_pattern(self.get_vocabulary(), text_buffer, pattern);

                    debug!(
                        "Last few chars of text stream: {:?}",
                        text_buffer
                            .chars()
                            .rev()
                            .take(15)
                            .collect::<Vec<_>>()
                            .iter()
                            .rev()
                            .collect::<String>()
                    );
                    trace!("naive mask: {mask:?}");

                    let next_token: String = self.sample_one_token(mask).to_owned();

                    debug!("Next token chosen: {next_token:?}");

                    if next_token == EOS_TOKEN {
                        info!("EOS token received returning earky: Generated {i} out of {max_tokens} max allowed");
                        break;
                    } else {
                        text_buffer.push_str(&next_token);
                    }
                }
            }
            MaskingAlgoState::IndexedFSM { ref map, mut fsm } => {
                info!("Using IndexedFSM algorithm");

                let current_state = &mut fsm.current_state;
                let fsm = fsm.inner;

                trace!("State index map {:?}", map);
                trace!("Current state {:?}", current_state);

                for i in 0..max_tokens {
                    let mut mask = Mask::zeros(self.vocabulary_size());

                    let vocab = self.get_vocabulary();

                    for (idx, bit) in mask.inner.iter_mut().enumerate() {
                        if map
                            .get(current_state)
                            .expect("key should exists for all states")
                            .contains(&vocab[idx])
                        {
                            *bit = 1;
                            trace!("state {:?} contains {}", current_state, &vocab[idx]);
                        } else {
                            *bit = 0;
                            trace!("state {:?} does not contain {}", current_state, &vocab[idx]);
                        }
                    }

                    debug!(
                        "Last few chars of text stream: {:?}",
                        text_buffer
                            .chars()
                            .rev()
                            .take(15)
                            .collect::<Vec<_>>()
                            .iter()
                            .rev()
                            .collect::<String>()
                    );
                    trace!("IndexedFSM mask: {mask:?}");

                    let next_token: String = self.sample_one_token(mask).to_owned();

                    debug!("Next token chosen: {next_token:?}");
                    
                    if next_token == EOS_TOKEN {
                        info!("EOS token received returning earky: Generated {i} out of {max_tokens} max allowed");
                        break;
                    } else {
                        text_buffer.push_str(&next_token);
                    }

                    // Advance FSM over the newly appended token
                    for &b in next_token.as_bytes() {
                        *current_state = fsm.next_state(*current_state, b);
                    }
                }
            }
        }
    }

    fn vocabulary_size(&self) -> usize {
        self.get_vocabulary().len()
    }
}

pub fn sample_model(
    model: &mut impl LangModel,
    max_tokens: usize,
    input_prompt: &str,
    masking_config: &MaskingAlgorithmConfig,
) -> Result<String> {
    let algo_state =
        MaskingAlgoState::from_config(masking_config, input_prompt, model.get_vocabulary());
    let mut text_buffer = input_prompt.to_owned();

    model.sample_multiple_tokens(max_tokens, &mut text_buffer, algo_state);

    Ok(text_buffer)
}

/// Inefficient implementation of a mask
fn naive_mask_from_pattern(vocabulary: &[Token], previous_samples: &str, pattern: &str) -> Mask {
    let mut mask = Mask::ones(vocabulary.len());

    for (i, tok) in vocabulary.iter().enumerate() {
        let mut possible_completion = previous_samples.to_owned();
        possible_completion.push_str(tok);

        let possible_input = Input::new(&possible_completion);

        let re = Regex::new(pattern).expect("Invalid regex");
        if re.is_match(possible_input) {
            mask.inner[i] = 1;
        } else {
            trace!("pattern {pattern} does not match completion {possible_completion}");
            mask.inner[i] = 0;
        }
    }

    mask
}

/// Precompute the states map on the given `vocabulary` of tokens.
///
/// **Algorithm 4** from the paper.
fn map_states_to_vocab(fsm: &dense::DFA<Vec<u32>>, vocabulary: &[Token]) -> IndexedFSMStates {
    let mut map = IndexedFSMStates::new();
    for state in fsm.tt.states() {
        map.insert(state.id(), Default::default());
    }
    for token in vocabulary {
        let subsequences = find_subsequences(fsm, token).unwrap();
        for sequence in subsequences {
            map.entry(sequence[0])
                .and_modify(|tokens| {
                    tokens.insert(token.clone());
                })
                .or_default();
        }
    }
    map
}

/// Find all sub-sequences of the Finite State Machine `fsm` that accept the `token` string.
///
/// **Algorithm 3** from the paper.
fn find_subsequences(fsm: &dense::DFA<Vec<u32>>, token: &Token) -> Result<Vec<StateSequence>> {
    let mut all_subseqs = vec![];

    'states_loop: for state in fsm.tt.states() {
        let mut state_sequence: StateSequence = vec![];
        let mut curr_state = state.id();

        // Only keep the states that read token[0]
        let peek_next = fsm.next_state(curr_state, token.as_bytes()[0]);
        if fsm.is_special_state(peek_next)
            && (fsm.is_dead_state(peek_next) || fsm.is_quit_state(peek_next))
        {
            continue 'states_loop;
        }

        // Walk the FSM
        for &b in token.as_bytes() {
            if fsm.is_special_state(curr_state)
                && (fsm.is_dead_state(curr_state) || fsm.is_quit_state(curr_state))
            {
                continue 'states_loop;
            }
            state_sequence.push(curr_state);

            curr_state = fsm.next_state(curr_state, b);
        }
        all_subseqs.push(state_sequence);
    }

    debug!("Token {token} has subsequences {all_subseqs:?}");
    Ok(all_subseqs)
}
