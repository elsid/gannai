extern crate gannai;
extern crate argparse;
extern crate rand;
extern crate rustc_serialize;

use std::io::Read;
use std::fs::File;

use rustc_serialize::json;

use gannai::neural_network::{ApplyConf, NetworkBuf, Network};

struct Args {
    conf: String,
    network: String,
}

fn main() {
    let mut args = Args {conf: String::new(), network: String::new()};
    parse_args(&mut args);
    let conf = make_conf(&args);
    let network_buf = make_network_buf(&args);
    apply(&conf, &network_buf.as_network());
}

fn parse_args(args: &mut Args) {
    use argparse::{ArgumentParser, Store};
    let mut parser = ArgumentParser::new();
    parser.set_description("Evolves neural network by using samples");
    parser.refer(&mut args.conf)
        .add_argument("conf", Store, "Path to conf json file").required();
    parser.refer(&mut args.network)
        .add_argument("network", Store, "Path to neural network json file").required();
    parser.parse_args_or_exit();
}

fn make_conf(args: &Args) -> ApplyConf {
    let mut data = String::new();
    File::open(&args.conf).unwrap().read_to_string(&mut data).unwrap();
    json::decode(&data).unwrap()
}

fn make_network_buf(args: &Args) -> NetworkBuf {
    let mut data = String::new();
    File::open(&args.network).unwrap().read_to_string(&mut data).unwrap();
    json::decode(&data).unwrap()
}

fn apply(conf: &ApplyConf, network: &Network) {
    use std::io::{BufRead, stdin};
    use gannai::neural_network::Apply;
    let application = network.apply(&conf);
    let file = stdin();
    for line in file.lock().lines() {
        let data = line.unwrap();
        let input: Vec<f64> = json::decode(&data).unwrap();
        let output = application.perform(&input[..]);
        println!("{}", json::encode(&output).unwrap());
    }
}
