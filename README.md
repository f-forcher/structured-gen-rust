# Efficient structured generation from Large Language Models
![example branch parameter](https://github.com/f-forcher/structured-gen-rust/actions/workflows/rust-tests.yml/badge.svg)

A proof-of-concept Rust implementation of the core algorithm from "Efficient structured generation for LLMs" paper.

# How to run
A small CLI application is provided to test the algorithm. You can either run it locally, or by using Docker.

### Locally
- If you haven't yet, [install rust](https://www.rust-lang.org/tools/install).
- Run the following command. The options are described below in the [Commands and options](#commands) section.
    ```
    cargo run --release -- <OPTIONS>
    ```

### Docker
- Build the image 
    ```
    docker build -t structured-gen .
    ```
- Run the app
    ```
    docker run -it --rm --name structured-gen structured-gen <OPTIONS>
    ```

## Commands and options
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
          - no-masking:  Do not perform structured generation, 
            mask will allow all tokens
          - naive:       Use naive `O(N)` pattern matching algorithm, i.e. check for each 
            token if the resulting completed output would still validate the pattern
          - indexed-fsm: The algorithm from the paper 
            [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702), 
            precomputing the token vocabulary with a hashmap from the pattern 
            FSM states to valid tokens. 
            The masking step is now O(1), indepentent of the current output sequence length

  -v, --vocab <VOCAB>...
          The model vocabulary as a space separated list of words. Example:
          
          -v A B 3 ...
          
          If not present, the default vocabulary 
          ["A", "3", ".", "42", "B", ".2", "1"] will be used.

  -i, --input <INPUT>
          The input prompt to the model. Keep in mind that the 
          whole text completion including the prompt, 
          must conform to the pattern. The default is an empty string
          
          [default: ]

  -p, --pattern <PATTERN>
          The regex pattern according to which the model output should conform. 
          Usually you want to anchor it at both ends, i.e. `^...$`. Default is 
          the float regex `^([0-9]*)?\.?[0-9]*$`
          
          [default: ^([0-9]*)?\.?[0-9]*$]

  -n, --n-tokens <N_TOKENS>
          The max amount of tokens to produce
          
          [default: 15]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

### Examples:
- Get help locally:
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