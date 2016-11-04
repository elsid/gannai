extern crate gannai;
extern crate argparse;
extern crate rand;
extern crate rustc_serialize;

use rustc_serialize::json;

use gannai::neural_network::{ApplyConf, Network};
use gannai::tools::common::{make_conf, make_network_buf};

struct Args {
    conf: String,
    network: String,
}

fn main() {
    let mut args = Args {conf: String::new(), network: String::new()};
    parse_args(&mut args);
    let conf = make_conf::<ApplyConf>(&args.conf);
    let network_buf = make_network_buf(&args.network);
    apply(&conf, &network_buf.as_network());
}

fn parse_args(args: &mut Args) {
    use argparse::{ArgumentParser, Store};
    let mut parser = ArgumentParser::new();
    parser.set_description("Applies neural network to input data");
    parser.refer(&mut args.conf)
        .add_argument("conf", Store, "Path to conf json file").required();
    parser.refer(&mut args.network)
        .add_argument("network", Store, "Path to neural network json file").required();
    parser.parse_args_or_exit();
}

fn apply(conf: &ApplyConf, network: &Network) {
    use std::io::{BufRead, stdin};
    use gannai::neural_network::Apply;
    let application = network.apply(&conf);
    let file = stdin();
    for line in file.lock().lines() {
        let data = line.unwrap();
        let input: Input = json::decode(&data).unwrap();
        let result_values = application.perform(&input.input[..]);
        let output = Output {input: input.input, output: input.output, result: result_values};
        println!("{}", json::encode(&output).unwrap());
    }
}

#[derive(RustcDecodable)]
struct Input {
    input: Vec<f64>,
    output: Option<Vec<f64>>,
}

#[derive(RustcEncodable)]
struct Output {
    input: Vec<f64>,
    output: Option<Vec<f64>>,
    result: Vec<f64>,
}
