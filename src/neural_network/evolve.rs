extern crate itertools;
extern crate rand;
extern crate rayon;

use self::itertools::Itertools;

use self::rand::Rng;

use super::train;
use super::train::Train;
use super::error::Error;
use super::common::{Node, Value};
use super::mutator::{Arc, Mutator};
use super::network::NetworkBuf;
use super::id_generator::IdGenerator;

pub struct Conf<'r, RngT: 'r + Rng> {
    pub train_conf: &'r train::Conf<'r>,
    pub rng: &'r mut RngT,
    pub node_id: &'r mut IdGenerator,
    pub population_size: usize,
    pub error: Value,
    pub iterations_count: usize,
}

pub trait Evolve {
    fn evolve<'r, RngT: 'r + Rng>(&self, conf: &'r mut Conf<'r, RngT>) -> Self;
}

impl Evolve for NetworkBuf {
    fn evolve<'r, RngT: 'r + Rng>(&self, conf: &'r mut Conf<'r, RngT>) -> Self {
        Mutator::from_network(&self.as_network()).evolve(conf).as_network_buf()
    }
}

impl Evolve for Mutator {
    fn evolve<'r, RngT: 'r + Rng>(&self, conf: &'r mut Conf<'r, RngT>) -> Self {
        Evolution::new(conf, self).perform()
    }
}

trait MutatorRandom {
    fn random_node<'r, RngT: 'r + Rng>(&self, rng: &mut RngT) -> Node;
    fn random_arc<'r, RngT: 'r + Rng>(&self, rng: &mut RngT) -> Arc;
}

impl MutatorRandom for Mutator {
    fn random_node<'r, RngT: 'r + Rng>(&self, rng: &mut RngT) -> Node {
        let nth = rng.gen_range(0, self.nodes().len());
        *self.nodes().nth(nth).unwrap()
    }

    fn random_arc<'r, RngT: 'r + Rng>(&self, rng: &mut RngT) -> Arc {
        let nth = rng.gen_range(0, self.arcs().len());
        self.arcs().nth(nth).unwrap().clone()
    }
}

struct Mutation {
    mutator: Mutator,
    error: Value,
}

struct Evolution<'c, 'f, RngT: 'c + Rng> {
    conf: &'c mut Conf<'c, RngT>,
    mutator: &'f Mutator,
    iterations_count: usize,
}

impl<'c, 'f, RngT: 'c + Rng> Evolution<'c, 'f, RngT> {
    pub fn new(conf: &'c mut Conf<'c, RngT>, mutator: &'f Mutator) -> Self {
        assert!(conf.population_size > 1);
        assert!(conf.population_size % 2 == 0);
        Evolution {conf: conf, mutator: mutator, iterations_count: 0}
    }

    pub fn perform(&mut self) -> Mutator {
        if !self.begin() {
            return self.mutator.clone();
        }
        let error = self.mutator.as_network_buf().as_network().error(self.conf.train_conf.error_conf);
        let population = (0..self.conf.population_size)
            .map(|_| Mutation {mutator: self.mutator.clone(), error: error})
            .collect::<Vec<_>>();
        let evolved = self.evolve(population);
        self.select_one(evolved)
    }

    fn evolve(&mut self, mut population: Vec<Mutation>) -> Vec<Mutation> {
        loop {
            let mutated = self.mutate(population);
            self.iterations_count += 1;
            if self.terminate(&mutated) {
                return mutated;
            }
            let propagated = self.propagate(mutated);
            population = self.select(propagated);
        }
    }

    fn mutate(&mut self, population: Vec<Mutation>) -> Vec<Mutation> {
        use self::rayon::prelude::{IntoParallelIterator, ParallelIterator, ExactParallelIterator};
        let ref mut node_id = self.conf.node_id;
        let ref mut rng = self.conf.rng;
        let ref train_conf = self.conf.train_conf;
        let mut result = Vec::new();
        population.into_iter()
            .map(|x| x.mutator)
            .map(|mut x| {
                for _ in 0..3 {
                    match rng.gen_range(0, 3) {
                        0 => {
                            let arc = x.random_arc(rng);
                            x.split(node_id, &arc);
                        },
                        1 => loop {
                            let src = x.random_node(rng);
                            let dst = x.random_node(rng);
                            if !x.graph().arcs().contains_key(&Arc(src, dst)) {
                                x.add_arc(src, dst, 1e-3);
                                break;
                            }
                        },
                        2 => {
                            let arc = x.random_arc(rng);
                            x.rm_arc(&arc);
                            x.rm_useless();
                        },
                        _ => (),
                    }
                }
                x
            })
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|x| {
                let mut network_buf = x.as_network_buf();
                let error = network_buf.as_network_mut().train(train_conf);
                Mutation {
                    mutator: Mutator::from_network(&network_buf.as_network()),
                    error: error,
                }
            })
            .collect_into(&mut result);
        result
    }

    fn begin(&self) -> bool {
        self.iterations_count < self.conf.iterations_count
        && self.mutator.as_network_buf().as_network().error(self.conf.train_conf.error_conf)
           > self.conf.error
    }

    fn terminate(&self, population: &Vec<Mutation>) -> bool {
        use std::io::{Write, stderr};
        writeln!(stderr(), "Evolve {}/{} iterations done",
               self.iterations_count, self.conf.iterations_count).unwrap();
        self.iterations_count >= self.conf.iterations_count
        || population.iter()
            .filter(|x| x.error <= self.conf.error)
            .count() > 0
    }

    fn propagate(&mut self, population: Vec<Mutation>) -> Vec<Mutation> {
        let mut new = {
            let mut a = population.iter().collect::<Vec<_>>();
            let mut b = population.iter().collect::<Vec<_>>();
            self.conf.rng.shuffle(&mut a[..]);
            self.conf.rng.shuffle(&mut b[..]);
            a.iter()
                .zip(b.iter())
                .map(|(l, r)| {
                    let mutator = l.mutator.union(&r.mutator);
                    let error = mutator.as_network_buf().as_network().error(self.conf.train_conf.error_conf);
                    Mutation {mutator: mutator, error: error}
                })
                .collect::<Vec<_>>()
        };
        new.extend(population.into_iter());
        new
    }

    fn select(&self, population: Vec<Mutation>) -> Vec<Mutation> {
        population.into_iter()
            .sorted_by(|l, r| l.error.partial_cmp(&r.error).unwrap())
            .into_iter()
            .take(self.conf.population_size)
            .collect()
    }

    fn select_one(&self, population: Vec<Mutation>) -> Mutator {
        population.into_iter()
            .sorted_by(|l, r| l.error.partial_cmp(&r.error).unwrap())
            .into_iter()
            .nth(0).unwrap().mutator
    }
}

#[test]
fn test_evolve_should_succeed() {
    extern crate rand;
    use std::collections::{BTreeSet, HashMap, HashSet};
    use self::rand::{XorShiftRng, SeedableRng};
    use neural_network::apply::{Conf as ApplyConf};
    use neural_network::error::{Conf as ErrorConf, Sample};
    use neural_network::train::{Conf as TrainConf};
    use neural_network::matrix::Matrix;
    use neural_network::network::Network;
    let mut weights_values = [
        0.0, 0.0, 0.0, 0.1, 0.1,
        0.0, 0.0, 0.0, 0.1, 0.1,
        0.0, 0.0, 0.0, 0.1, 0.1,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let inputs = [0, 1, 2].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [3, 4].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(5, &mut weights_values);
    let nodes = (0..5).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {
        inputs: &inputs,
        outputs: &outputs,
        weights: weights,
        nodes: &nodes
    };
    let mutator = Mutator::from_network(&network);
    let apply_conf = ApplyConf {
        threshold: 1e-4,
    };
    let samples = [
        Sample {input: &[0.6, 0.7, 0.8], output: &[0.5, 0.4]},
        Sample {input: &[0.3, 0.4, 0.5], output: &[0.3, 0.4]},
    ];
    let error_conf = ErrorConf {
        apply_conf: &apply_conf,
        samples: &samples,
    };
    let train_conf = TrainConf {
        error_conf: &error_conf,
        max_function_calls_count: 111,
    };
    let mut rng = XorShiftRng::new_unseeded();
    rng.reseed([1, 1, 1, 1]);
    let mut node_id = IdGenerator::new(0);
    let mut conf = Conf {
        train_conf: &train_conf,
        rng: &mut rng,
        node_id: &mut node_id,
        population_size: 2,
        error: 1e-3,
        iterations_count: 2,
    };
    let evolved = mutator.evolve(&mut conf);
    let evolved_network_buf = evolved.as_network_buf();
    let evolved_network = evolved_network_buf.as_network();
    let result: &[f64] = evolved_network.weights.values();
    assert_eq!(result, &[
        0.03541497627352951 , 0.024044759053380473, 0.024984884403041474, 0.2074014127129306  , 0.0084995228018026  , 0.5931675324958909, 0.0                 , 0.0,
        0.03216048430891172 , 0.03283758771509236 , 0.025092654918090918, 0.22227547676945245 , 0.1783386201627317  , 0.0               , 0.02260390906455018 , 0.0,
        0.022630703472141002, 0.022590572563856454, 0.0                 , 0.15469676456743142 , 0.16814293629478178 , 0.0               , 0.023389761417945736, 0.1516660210553475,
        0.031646940802308635, 0.02717475673890699 , 0.02496441865360037 , 0.09643870193849251 , 0.057538757850797775, 0.0               , 0.02279107696598028 , 0.0,
        0.02281617356785006 , 0.022793461846504364, 0.022826128788326887, 0.02253492937610908 , 0.022562257735017913, 0.0               , 0.02283095908060125 , 0.0,
        0.0                 , 0.0                 , 0.0                 , 0.0                 , 0.0                 , 0.0               , 0.5931683531387447  , 0.0,
        0.0                 , 0.023113295196451642, 0.022918354550158854, 0.022613155400683566, 0.3516975811633756  , 0.0               , 0.022946044823162547, 0.0,
        0.0                 , 0.0                 , 0.1516660210553475  , 0.0                 , 0.0                 , 0.0               , 0.0                 , 0.0,
    ] as &[f64]);
    assert_eq!(0.6795650560090591, mutator.as_network_buf().as_network().error(&error_conf));
    assert_eq!(0.18891967136068427, evolved_network.error(&error_conf));
}

#[test]
fn test_evolve_zero_iterations_should_do_nothing() {
    extern crate rand;
    use std::collections::{BTreeSet, HashMap, HashSet};
    use self::rand::{XorShiftRng, SeedableRng};
    use neural_network::apply::{Conf as ApplyConf};
    use neural_network::error::{Conf as ErrorConf, Sample};
    use neural_network::train::{Conf as TrainConf};
    use neural_network::matrix::Matrix;
    use neural_network::network::Network;
    let mut weights_values = [
        0.0, 0.0, 0.0, 0.1, 0.1,
        0.0, 0.0, 0.0, 0.1, 0.1,
        0.0, 0.0, 0.0, 0.1, 0.1,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let inputs = [0, 1, 2].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [3, 4].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(5, &mut weights_values);
    let nodes = (0..5).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {
        inputs: &inputs,
        outputs: &outputs,
        weights: weights,
        nodes: &nodes
    };
    let mutator = Mutator::from_network(&network);
    let apply_conf = ApplyConf {
        threshold: 1e-4,
    };
    let samples = [
        Sample {input: &[0.6, 0.7, 0.8], output: &[0.5, 0.4]},
        Sample {input: &[0.3, 0.4, 0.5], output: &[0.3, 0.4]},
    ];
    let error_conf = ErrorConf {
        apply_conf: &apply_conf,
        samples: &samples,
    };
    let train_conf = TrainConf {
        error_conf: &error_conf,
        max_function_calls_count: 111,
    };
    let mut rng = XorShiftRng::new_unseeded();
    rng.reseed([1, 1, 1, 1]);
    let mut node_id = IdGenerator::new(0);
    let mut conf = Conf {
        train_conf: &train_conf,
        rng: &mut rng,
        node_id: &mut node_id,
        population_size: 2,
        error: 1e-3,
        iterations_count: 0,
    };
    let evolved = mutator.evolve(&mut conf);
    let evolved_network_buf = evolved.as_network_buf();
    let evolved_network = evolved_network_buf.as_network();
    let result: &[f64] = evolved_network.weights.values();
    assert_eq!(result, &[
        0.0, 0.0, 0.0, 0.1, 0.1,
        0.0, 0.0, 0.0, 0.1, 0.1,
        0.0, 0.0, 0.0, 0.1, 0.1,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ] as &[f64]);
}
