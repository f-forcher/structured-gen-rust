use anyhow::Result;
use regex_automata::{
    dfa::{dense, Automaton},
    Anchored, Input,
};
use structured_gen_rust::utils::{DeterministicModel, LangModel, MaskingAlgo};

fn main() -> Result<()> {
    let dfa = dense::DFA::new(r"[0-9]*\.?[0-9]*")?;
    let haystack = "1.34";

    // The start state is determined by inspecting the position and the
    // initial bytes of the haystack.
    let mut state = dfa.start_state_forward(&Input::new(haystack).anchored(Anchored::Yes))?;
    // Walk all the bytes in the haystack.
    for &b in haystack.as_bytes().iter() {
        state = dfa.next_state(state, b);
    }

    let states_num = dfa.tt.len();

    println!("There are {} states in this dfa", states_num);

    let states_ids: Vec<_> = dfa.tt.states().map(|state| state.id()).collect();

    println!("Their IDs are {:?}", states_ids);

    // DFAs in this crate require an explicit
    // end-of-input transition if a search reaches
    // the end of a haystack.
    state = dfa.next_eoi_state(state);
    assert!(dfa.is_match_state(state));

    /////// Mock LLM test with pattern
    let vocabulary: Vec<String> = ["A", ".", "42", "B", ".2", "1"]
        .into_iter()
        .map(|s| s.to_owned())
        .collect();
    let mut determ = DeterministicModel::new(vocabulary);

    let mut previous_samples = String::new();
    let test_cyclic_out = determ.sample_n_tokens(
        15,
        &mut previous_samples,
        MaskingAlgo::Naive { pattern: None },
    );
    println!("Test cyclic mockllm: {:?}", test_cyclic_out);

    /////// Mock LLM test
    let vocabulary: Vec<String> = ["A", "3", ".", "42", "B", ".2", "1"]
        .into_iter()
        .map(|s| s.to_owned())
        .collect();
    let mut determ = DeterministicModel::new(vocabulary);

    let mut previous_samples2 = String::new();

    let temp = &String::from(r"^([0-9]*)?\.?[0-9]*$");
    let pattern = Some(temp);
    let test_cyclic_out2 =
        determ.sample_n_tokens(15, &mut previous_samples2, MaskingAlgo::Naive { pattern });
    println!("Test cyclic mockllm: {:?}", test_cyclic_out2);

    Ok(())
}
