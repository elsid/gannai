extern crate gannai;
extern crate dot;
extern crate rand;

use std::fs::File;

use rand::{XorShiftRng, SeedableRng};

use gannai::neural_network::{
    Apply,
    ApplyConf,
    ErrorConf,
    Error,
    Mutator,
    Evolve,
    EvolveConf,
    IdGenerator,
    Sample,
    Train,
    TrainConf,
};

fn main() {
    let mut node_id = IdGenerator::new(0);
    let mutator = Mutator::new(&mut node_id, 4, 3, 1e-3);
    let apply_conf = ApplyConf {
        group_size: 1000,
        threshold: 1e-4,
    };
    let samples = [
        Sample {input: &[0.6, 0.7, 0.8, 0.9], output: &[0.5, 0.4, 0.3]},
    ];
    let mut network_buf = mutator.as_network_buf();
    let error_conf = ErrorConf {
        apply_conf: &apply_conf,
        samples: &samples,
    };
    println!("initial error: {}", network_buf.as_network().error(&error_conf));
    for sample in samples.iter() {
        println!("initial apply {:?}",
                 network_buf.as_network().apply(&apply_conf).perform(sample.input));
    }
    {
        let mut f = File::create("initial.dot").unwrap();
        dot::render(Mutator::from_network(&network_buf.as_network()).graph(), &mut f).unwrap();
    }
    let train_conf = TrainConf {
        error_conf: &error_conf,
        max_function_calls_count: 1000,
    };
    println!("trained error: {}", network_buf.as_network_mut().train(&train_conf));
    for sample in samples.iter() {
        println!("trained apply {:?}",
                 network_buf.as_network().apply(&apply_conf).perform(sample.input));
    }
    {
        let mut f = File::create("trained.dot").unwrap();
        dot::render(Mutator::from_network(&network_buf.as_network()).graph(), &mut f).unwrap();
    }
    let mut rng = XorShiftRng::new_unseeded();
    rng.reseed([1, 1, 1, 1]);
    let mut evolve_conf = EvolveConf {
        train_conf: &train_conf,
        rng: &mut rng,
        node_id: &mut node_id,
        population_size: 4,
        error: 1e-3,
        iterations_count: 3,
    };
    let evolved = mutator.evolve(&mut evolve_conf);
    let evolved_network_buf = evolved.as_network_buf();
    println!("evolved error: {}", evolved_network_buf.as_network().error(&error_conf));
    for sample in samples.iter() {
        println!("evolved apply {:?}",
                 evolved_network_buf.as_network().apply(&apply_conf).perform(sample.input));
    }
    {
        let mut f = File::create("evolved.dot").unwrap();
        dot::render(evolved.graph(), &mut f).unwrap();
    }
}
