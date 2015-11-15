extern crate argparse;
extern crate gannai;
extern crate rustc_serialize;

use gannai::neural_network::NetworkBuf;

struct Args {
    input_nodes_count: usize,
    output_nodes_count: usize,
    initial_weight: f64,
}

fn main() {
    use rustc_serialize::json;
    let mut args = Args {input_nodes_count: 1, output_nodes_count: 1, initial_weight: 1e-3};
    parse_args(&mut args);
    let network = generate(&args);
    println!("{}", json::encode(&network).unwrap());
}

fn parse_args(args: &mut Args) {
    use argparse::{ArgumentParser, Store};
    let mut parser = ArgumentParser::new();
    parser.set_description("Generates neural network");
    parser.refer(&mut args.input_nodes_count)
        .add_argument("input", Store, "Number of input nodes").required();
    parser.refer(&mut args.output_nodes_count)
        .add_argument("output", Store, "Number of output nodes").required();
    parser.refer(&mut args.initial_weight)
        .add_argument("weight", Store, "Initial weight of connections");
    parser.parse_args_or_exit();
}

fn generate(args: &Args) -> NetworkBuf {
    use gannai::neural_network::{Mutator, IdGenerator};
    let mut node_id = IdGenerator::new(0);
    Mutator::new(
        &mut node_id,
        args.input_nodes_count,
        args.output_nodes_count,
        args.initial_weight)
        .as_network_buf()
}
