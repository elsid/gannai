extern crate gannai;
extern crate argparse;
extern crate dot;
extern crate rustc_serialize;

use gannai::neural_network::NetworkBuf;

struct Args {
    network: String,
}

fn main() {
    use std::io::stdout;
    use gannai::neural_network::Mutator;
    let mut args = Args {network: String::new()};
    parse_args(&mut args);
    let network_buf = make_network_buf(&args);
    let mut output = stdout();
    dot::render(Mutator::from_network(&network_buf.as_network()).graph(), &mut output).unwrap();
}

fn parse_args(args: &mut Args) {
    use argparse::{ArgumentParser, Store};
    let mut parser = ArgumentParser::new();
    parser.set_description("Converts network to .dot format");
    parser.refer(&mut args.network)
        .add_argument("network", Store, "Path to neural network json file (default is stdin)");
    parser.parse_args_or_exit();
}

fn make_network_buf(args: &Args) -> NetworkBuf {
    use std::io::{Read, stdin};
    use std::fs::File;
    use rustc_serialize::json;
    let mut data = String::new();
    if args.network.is_empty() {
        stdin().read_to_string(&mut data).unwrap();
    } else {
        File::open(&args.network).unwrap().read_to_string(&mut data).unwrap();
    }
    json::decode(&data).unwrap()
}
