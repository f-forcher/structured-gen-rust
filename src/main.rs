use regex_automata::{dfa::{dense, Automaton}, Anchored, Input};
use anyhow::Result;

fn main() -> Result<()>  {

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

    let states_ids: Vec<_> = dfa.tt.states()
        .map(|state| state.id()).collect();

    println!("Their IDs are {:?}", states_ids);

    // DFAs in this crate require an explicit
    // end-of-input transition if a search reaches
    // the end of a haystack.
    state = dfa.next_eoi_state(state);
    assert!(dfa.is_match_state(state));

    return Ok(())
}
