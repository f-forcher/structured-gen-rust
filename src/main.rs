use anyhow::Result;

use log::trace;
use rand::thread_rng;
use structured_gen_rust::{
    sample_model,
    util::{generate_dict, DeterministicModel, RandomSampleModel},
    MaskingAlgorithmConfig,
};

use clap::{Parser, ValueEnum};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Name of language model to use. At the moment only "mock" options are available.
    #[arg(short, long, value_enum, default_value_t = ModelSelector::RandomSample)]
    model: ModelSelector,

    #[arg(short, long, value_enum, default_value_t = AlgorithmSelector::IndexedFSM)]
    algo: AlgorithmSelector,

    /// The model vocabulary as a space separated list of words. Example:
    ///
    /// -v A B 3 ...
    ///
    /// If not present, the default vocabulary ["A", "3", ".", "42", "B", ".2", "1"]
    /// will be used.
    ///
    /// If set, it overrides the --gen-vocab option.
    #[arg(short, long, value_parser, num_args = 1.., value_delimiter = ' ')]
    vocab: Option<Vec<String>>,

    /// The input prompt to the model. Keep in mind that the whole text completion including
    /// the prompt, must conform to the pattern. The default is an empty string.
    #[arg(short, long, default_value_t = String::from(r""))]
    input: String,

    /// The regex pattern according to which the model output should conform.
    /// Usually you want to anchor it at both ends, i.e. `^...$`.
    /// Default is the float regex `^([0-9]*)?\.?[0-9]*$`
    #[arg(short, long, default_value_t = String::from(r"^([0-9]*)?\.?[0-9]*$"))]
    pattern: String,

    /// The max amount of tokens to produce
    #[arg(short, long, default_value_t = 15)]
    n_tokens: usize,

    /// You can set this to generate a vocabulary with `usize` tokens inside.
    ///
    /// The dictionary consists of the
    /// single chars `a-z A-Z 0-9 . : , ! ?` and every multiple char cartesian
    /// product combination of these, generating up to `gen_vocab` tokens.
    ///
    /// If neither this or `--vocab` is set, the default vocabulary will be used
    /// (see `--vocab` for more details).
    #[arg(short, long)]
    gen_vocab: Option<usize>,
}

fn default_small_dict() -> Vec<String> {
    let tokens = vec!["A", "3", ".", "42", "B", ".2", "1"];
    let vocabulary: Vec<String> = tokens.into_iter().map(|s| s.to_owned()).collect();
    vocabulary
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ModelSelector {
    /// Simple mock language model that just
    /// iterates cyclically over its vocabulary.
    Deterministic,
    /// Model that randomly samples from a weighted list of tokens.
    RandomSample,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
/// Select the token masking algorithm.
pub enum AlgorithmSelector {
    /// Do not perform structured generation, mask will allow all tokens.
    NoMasking,

    /// Use naive `O(N)` pattern matching algorithm, i.e. check for each token
    /// if the resulting completed output would still validate the pattern.
    Naive,

    /// The algorithm from the paper
    /// [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702),
    /// precomputing the token vocabulary with a hashmap from the pattern FSM states
    /// to valid tokens. The masking step is now O(1), indepentent
    /// of the current output sequence length.
    IndexedFSM,
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    let vocabulary = match (cli.vocab, cli.gen_vocab) {
        (Some(vocab), Some(_)) => vocab,
        (Some(vocab), None) => vocab,
        (None, None) => default_small_dict(),
        (None, Some(gen_tokens)) => generate_dict(gen_tokens),
    };

    let input_prompt = cli.input;
    let pattern = &cli.pattern[..];
    let max_tokens = cli.n_tokens;

    let algo = match cli.algo {
        AlgorithmSelector::NoMasking => MaskingAlgorithmConfig::NoMasking,
        AlgorithmSelector::Naive => MaskingAlgorithmConfig::Naive(pattern),
        AlgorithmSelector::IndexedFSM => MaskingAlgorithmConfig::IndexedFSM(pattern),
    };

    let output = match cli.model {
        ModelSelector::Deterministic => {
            let mut model = DeterministicModel::new(&vocabulary);

            sample_model(&mut model, max_tokens, &input_prompt, &algo)?
        }
        ModelSelector::RandomSample => {
            let rng = thread_rng();
            let mut model = RandomSampleModel::new(&vocabulary, rng);

            sample_model(&mut model, max_tokens, &input_prompt, &algo)?
        }
    };

    trace!("Vocabulary: {vocabulary:?}");

    println!("Model output: \"{output}\"");

    Ok(())
}
