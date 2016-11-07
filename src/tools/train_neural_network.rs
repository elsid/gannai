extern crate gannai;
extern crate argparse;
extern crate rustc_serialize;

use rustc_serialize::json;

use gannai::neural_network::NetworkBuf;
use gannai::tools::common::{Sample, make_conf, make_samples, make_network_buf};

struct Args {
    conf: String,
    samples: String,
    network: String,
}

#[derive(RustcDecodable)]
struct Conf {
    threshold: f64,
    max_function_calls_count: usize,
}

fn main() {
    let mut args = Args {conf: String::new(), samples: String::new(), network: String::new()};
    parse_args(&mut args);
    let conf = make_conf::<Conf>(&args.conf);
    let mut network_buf = make_network_buf(&args.network);
    let samples = make_samples(&args.samples);
    train(&conf, &samples, &mut network_buf);
    println!("{}", json::encode(&network_buf).unwrap());
}

fn parse_args(args: &mut Args) {
    use argparse::{ArgumentParser, Store};
    let mut parser = ArgumentParser::new();
    parser.set_description("Trains neural network by using samples");
    parser.refer(&mut args.conf)
        .add_argument("conf", Store, "Path to conf json file").required();
    parser.refer(&mut args.network)
        .add_argument("network", Store, "Path to neural network json file").required();
    parser.refer(&mut args.samples)
        .add_argument("samples", Store, "Path to samples json file (default is stdin)");
    parser.parse_args_or_exit();
}

fn train(conf: &Conf, src_samples: &[Sample], network_buf: &mut NetworkBuf) {
    use gannai::neural_network::{
        ApplyConf,
        ErrorConf,
        Sample,
        Train,
        TrainConf,
    };
    let apply_conf = ApplyConf {
        threshold: conf.threshold,
    };
    let samples: Vec<Sample> = src_samples.iter()
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
