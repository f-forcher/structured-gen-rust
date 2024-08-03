# Efficient Guided Generation for Large Language Models
![example branch parameter](https://github.com/f-forcher/structured-gen-rust/actions/workflows/rust-tests.yml/badge.svg)

A proof-of-concept Rust implementation of the algorithm from ["Efficient Guided generation for LLMs"](https://arxiv.org/abs/2307.09702) paper.

# How to run
A small CLI application is provided to test the algorithm. You can either run it locally, inside a devcontainer, 
or by using Docker directly.

### Local
- If you haven't yet, [install rust](https://www.rust-lang.org/tools/install).
- Run the following command. Use the `--help` option to see a list of possible args to configure or see below
  the [Arguments](#arguments) section.
    ```
    cargo run --release -- <OPTIONS>
    ```

### Devcontainer
- If [devcontainer](https://code.visualstudio.com/docs/devcontainers/devcontainer-cli) is installed in your IDE, you can
  open this repository in the container, wait for the image to build, then just run `cargo` commands as shown in [Local](#local) section.

### Docker
- Build the image 
    ```
    docker build -t structured-gen .
    ```
- Run the app
    ```
    docker run -it --rm --name structured-gen structured-gen <OPTIONS>
    ```

# Arguments
You can use the `--help` option to get a list of the possible commands:

```
Usage: structured-gen-rust [OPTIONS]

Options:
  -m, --model <MODEL>
          Name of language model to use. At the moment only "mock" options are available

          [default: random-sample]

          Possible values:
          - deterministic: Simple mock language model that just iterates cyclically over its vocabulary
          - random-sample: Model that randomly samples from a weighted list of tokens

  -a, --algo <ALGO>
          [default: indexed-fsm]

          Possible values:
          - no-masking:  Do not perform structured generation, mask will allow all tokens
          - naive:       Use naive `O(N)` pattern matching algorithm, i.e. check for each token if the 
          resulting completed output would still validate the pattern
          - indexed-fsm: The algorithm from the paper [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702), 
          precomputing the token vocabulary with a hashmap from the pattern FSM states to valid tokens. 
          The masking step is now O(1), indepentent of the current output sequence length

  -v, --vocab <VOCAB>...
          The model vocabulary as a space separated list of words. Example:

          -v A B 3 ...

          If not present, the default vocabulary ["A", "3", ".", "42", "B", ".2", "1"] will be used.

          If set, it overrides the --gen-vocab option.

  -i, --input <INPUT>
          The input prompt to the model. Keep in mind that the whole text completion including the 
          prompt, must conform to the pattern. The default is an empty string

          [default: ]

  -p, --pattern <PATTERN>
          The regex pattern to which the model output should conform. Usually you want to anchor 
          it at both ends, i.e. `^...$`. Default is the float regex `^([0-9]*)?\.?[0-9]*$`

          [default: ^([0-9]*)?\.?[0-9]*$]

  -n, --n-tokens <N_TOKENS>
          The max amount of tokens to produce

          [default: 15]

  -g, --gen-vocab <GEN_VOCAB>
          You can set this to generate a vocabulary with `usize` tokens inside.

          The dictionary consists of the single chars `a-z A-Z 0-9 . : , ! ?` and every multiple 
          char cartesian product combination of these, generating up to `gen_vocab` tokens.

          If neither this or `--vocab` is set, the default vocabulary will be used (see `--vocab` for more details).

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

## Examples:
- Get help locally (or in a devcontainer shell):
    ```
    cargo run --release -- --help
    ```
- Get help with docker:

    ```
    docker build -t structured-gen .

    docker run -it --rm --name structured-gen structured-gen --help
    ```
- Time the generation of up to 10000 tokens using the deterministic mock language model and the naive algorithm, with a custom token vocabulary. This takes about 1.5 seconds on my machine:

    ```
    time cargo run --release -- -m deterministic -a naive -n 10000 -v A B 3 . 45
    ```

- Now compare with the same command but using the efficient algorithm `indexed-fsm`. Now it is just
about 0.04 seconds (on my machine):

    ```
    time cargo run --release -- -m deterministic -a indexed-fsm -n 10000 -v A B 3 . 45
    ```

# Benchmarks
Benchmarks can be run by using the command `cargo bench`. 
In `target/criterion/report/index.html` an HTML plot will be generated
containing several plots and statistics.

# Logging and debugging
Debug logs can be enabled by running with a debug profile (i.e. `cargo run` without the `--release` flag) and setting the `RUST_LOG` env variable.

The log levels are, in order: `error`, `warn`, `info`, `debug`, `trace`.
Set the `RUST_LOG` env variable to one of them to enable more (or
less) detailed logging. Example:

```
RUST_LOG=debug cargo run -- -m deterministic -a indexed-fsm -g 500 -n 500
```

# Notes
## `regex` fork
The `regex-automata` public API does not expose the internal states
of the automata, so a [fork](https://github.com/f-forcher/regex/tree/expose-state-iter) of the Rust stdlib `regex` repo has been made 
and its internal `State` API exposed.

## Known issues
Using very large dictionaries may result in failure to produce the right output. The issue seems to be deterministic and independent of the
specific struct-gen algo used, only depending on the dictionary size. It could be connected to the `regex` crate internal implementation of FSM states.
