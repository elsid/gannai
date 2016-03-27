extern crate gannai;
extern crate argparse;
extern crate rustc_serialize;

use rustc_serialize::json;

use gannai::neural_network::NetworkBuf;

struct Args {
    conf: String,
    network: String,
}

#[derive(RustcDecodable)]
struct Sample {
    input: Vec<f64>,
    output: Vec<f64>,
}

#[derive(RustcDecodable)]
struct Conf {
    group_size: usize,
    threshold: f64,
    samples: Vec<Sample>,
    max_function_calls_count: usize,
}

fn main() {
    let mut args = Args {conf: String::new(), network: String::new()};
    parse_args(&mut args);
    let conf = make_conf(&args);
    let mut network_buf = make_network_buf(&args);
    train(&conf, &mut network_buf);
    println!("{}", json::encode(&network_buf).unwrap());
}

fn parse_args(args: &mut Args) {
    use argparse::{ArgumentParser, Store};
    let mut parser = ArgumentParser::new();
    parser.set_description("Trains neural network by using samples");
    parser.refer(&mut args.conf)
        .add_argument("conf", Store, "Path to conf json file").required();
    parser.refer(&mut args.network)
        .add_argument("network", Store, "Path to neural network json file (default is stdin)");
    parser.parse_args_or_exit();
}

fn make_conf(args: &Args) -> Conf {
    use std::io::Read;
    use std::fs::File;
    let mut data = String::new();
    File::open(&args.conf).unwrap().read_to_string(&mut data).unwrap();
    json::decode(&data).unwrap()
}

fn make_network_buf(args: &Args) -> NetworkBuf {
    use std::io::{Read, stdin};
    use std::fs::File;
    let mut data = String::new();
    if args.network.is_empty() {
        stdin().read_to_string(&mut data).unwrap();
    } else {
        File::open(&args.network).unwrap().read_to_string(&mut data).unwrap();
    }
    json::decode(&data).unwrap()
}

fn train(conf: &Conf, network_buf: &mut NetworkBuf) {
    use gannai::neural_network::{
        ApplyConf,
        ErrorConf,
        Sample,
        Train,
        TrainConf,
    };
    let apply_conf = ApplyConf {
        group_size: conf.group_size,
        threshold: conf.threshold,
    };
    let samples: Vec<Sample> = conf.samples.iter()
        .map(|x| Sample {input: &x.input[..], output: &x.output[..]})
        .collect::<_>();
    let error_conf = ErrorConf {
        apply_conf: &apply_conf,
        samples: &samples[..],
    };
    let train_conf = TrainConf {
        error_conf: &error_conf,
        max_function_calls_count: conf.max_function_calls_count,
    };
    network_buf.as_network_mut().train(&train_conf);
}
