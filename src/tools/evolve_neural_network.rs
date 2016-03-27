extern crate gannai;
extern crate argparse;
extern crate rand;
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
    error: f64,
    population_size: usize,
    iterations_count: usize,
}

fn main() {
    let mut args = Args {conf: String::new(), network: String::new()};
    parse_args(&mut args);
    let conf = make_conf(&args);
    let network_buf = make_network_buf(&args);
    let evolved_network_buf = evolve(&conf, network_buf);
    println!("{}", json::encode(&evolved_network_buf).unwrap());
}

fn parse_args(args: &mut Args) {
    use argparse::{ArgumentParser, Store};
    let mut parser = ArgumentParser::new();
    parser.set_description("Evolves neural network by using samples");
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

fn evolve(conf: &Conf, network_buf: NetworkBuf) -> NetworkBuf {
    use rand::{XorShiftRng, SeedableRng};
    use gannai::neural_network::{
        ApplyConf,
        ErrorConf,
        Mutator,
        Evolve,
        EvolveConf,
        IdGenerator,
        Sample,
        TrainConf,
    };
    let mut node_id = IdGenerator::new(0);
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
    let mut rng = XorShiftRng::new_unseeded();
    rng.reseed([1, 1, 1, 1]);
    let mut evolve_conf = EvolveConf {
        train_conf: &train_conf,
        rng: &mut rng,
        node_id: &mut node_id,
        population_size: conf.population_size,
        error: conf.error,
        iterations_count: conf.iterations_count,
    };
    Mutator::from_network(&network_buf.as_network()).evolve(&mut evolve_conf).as_network_buf()
}
