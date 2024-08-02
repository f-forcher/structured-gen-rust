use anyhow::Result;

use structured_gen_rust::{sample_model, util::DeterministicModel, MaskingAlgorithmConfig};

fn main() -> Result<()> {
    //let float_pattern = r"^([0-9]*)?\.?[0-9]*$";
    //  TODO Example of wrong pattern that will never match vocab tokens {"A", "B"}
    // let simple_pattern = r"^A+B+$";
    //let simple_pattern = r"^A?B*$";

    let simple_pattern = r"^A{3,5}B{0,5}$";
    let input_prompt = String::from("AAAAA");
    let vocabulary: Vec<String> = ["A", "3", ".", "42", "B", ".2", "1"]
        .into_iter()
        .map(|s| s.to_owned())
        .collect();
    let mut determ = DeterministicModel::new(vocabulary);

    let out = sample_model(
        &mut determ,
        15,
        &input_prompt,
        &MaskingAlgorithmConfig::Naive(simple_pattern), //MaskingAlgorithmConfig::IndexedFSM(simple_pattern),
    )
    .unwrap();
    println!("Test new fn with pattern: {:?}", out);
    Ok(())
}
