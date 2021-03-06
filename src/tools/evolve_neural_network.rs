extern crate gannai;
extern crate argparse;
extern crate rand;
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
    group_size: usize,
    threshold: f64,
    max_function_calls_count: usize,
    error: f64,
    population_size: usize,
    iterations_count: usize,
}

fn main() {
    let mut args = Args {conf: String::new(), samples: String::new(), network: String::new()};
    parse_args(&mut args);
    let conf = make_conf(&args.conf);
    let network_buf = make_network_buf(&args.network);
    let samples = make_samples(&args.samples);
    let evolved_network_buf = evolve(&conf, &samples, network_buf);
    println!("{}", json::encode(&evolved_network_buf).unwrap());
}

fn parse_args(args: &mut Args) {
    use argparse::{ArgumentParser, Store};
    let mut parser = ArgumentParser::new();
    parser.set_description("Evolves neural network by using samples");
    parser.refer(&mut args.conf)
        .add_argument("conf", Store, "Path to conf json file").required();
    parser.refer(&mut args.network)
        .add_argument("network", Store, "Path to neural network json file").required();
    parser.refer(&mut args.samples)
        .add_argument("samples", Store, "Path to samples json file (default is stdin)");
    parser.parse_args_or_exit();
}

fn evolve(conf: &Conf, src_samples: &[Sample], network_buf: NetworkBuf) -> NetworkBuf {
    use std::io::{Write, stderr};
    use rand::{XorShiftRng, SeedableRng};
    use gannai::neural_network::{
        ApplyConf,
        Error,
        ErrorConf,
        Evolve,
        EvolveConf,
        IdGenerator,
        Mutator,
        Sample,
        TrainConf,
    };
    let mut node_id = IdGenerator::new(0);
    let apply_conf = ApplyConf {
        group_size: conf.group_size,
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
    let network = network_buf.as_network();
    let initial_error = network.error(&error_conf);
    writeln!(stderr(), "Initial error: {}", initial_error).unwrap();
    let result = Mutator::from_network(&network).evolve(&mut evolve_conf).as_network_buf();
    let final_error = result.as_network().error(&error_conf);
    writeln!(stderr(), "Final error: {} (improved by {} times)", final_error, initial_error / final_error).unwrap();
    result
}
