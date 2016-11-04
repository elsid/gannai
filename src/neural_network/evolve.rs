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
        let population = (0..self.conf.population_size)
            .map(|_| self.mutator.clone()).collect::<Vec<_>>();
        let evolved = self.evolve(population);
        self.select_one(evolved)
    }

    fn evolve(&mut self, mut population: Vec<Mutator>) -> Vec<Mutator> {
        loop {
            let mutated = self.mutate(population);
            self.iterations_count += 1;
            if self.terminate(&mutated[..]) {
                return mutated;
            }
            let propagated = self.propagate(mutated);
            population = self.select(propagated);
        }
    }

    fn mutate(&mut self, population: Vec<Mutator>) -> Vec<Mutator> {
        use self::rayon::prelude::{IntoParallelIterator, ParallelIterator, ExactParallelIterator};
        let ref mut node_id = self.conf.node_id;
        let ref mut rng = self.conf.rng;
        let ref train_conf = self.conf.train_conf;
        let mut result = Vec::new();
        population.into_iter()
            .map(|mut x| {
                for _ in 0..3 {
                    match rng.gen_range(0, 3) {
                        0 => {
                            let arc = x.random_arc(rng);
                            x.split(node_id, &arc);
                        },
                        1 => {
                            let src = x.random_node(rng);
                            let dst = x.random_node(rng);
                            x.add_arc(src, dst, 0.1);
                        },
                        2 => {
                            let arc = x.random_arc(rng);
                            x.rm_arc(&arc);
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
                network_buf.as_network_mut().train(train_conf);
                Mutator::from_network(&network_buf.as_network())
            })
            .collect_into(&mut result);
        result
    }

    fn begin(&self) -> bool {
        self.iterations_count < self.conf.iterations_count
        && self.mutator.as_network_buf().as_network().error(self.conf.train_conf.error_conf)
           > self.conf.error
    }

    fn terminate(&self, population: &[Mutator]) -> bool {
        use std::io::{Write, stderr};
        write!(&mut stderr(), "Evolve {}/{} iterations done\n",
               self.iterations_count, self.conf.iterations_count).unwrap();
        self.iterations_count >= self.conf.iterations_count
        || population.iter()
            .map(|x| x.as_network_buf().as_network().error(self.conf.train_conf.error_conf))
            .filter(|&x| x <= self.conf.error)
            .count() > 0
    }

    fn propagate(&mut self, population: Vec<Mutator>) -> Vec<Mutator> {
        let mut new = {
            let mut a = population.iter().collect::<Vec<_>>();
            let mut b = population.iter().collect::<Vec<_>>();
            self.conf.rng.shuffle(&mut a[..]);
            self.conf.rng.shuffle(&mut b[..]);
            a.iter().zip(b.iter()).map(|(l, r)| l.union(r)).collect::<Vec<_>>()
        };
        new.extend(population.into_iter());
        new
    }

    fn select(&self, population: Vec<Mutator>) -> Vec<Mutator> {
        population.into_iter()
            .map(|x| (x.as_network_buf().as_network().error(self.conf.train_conf.error_conf), x))
            .sorted_by(|l, r| l.0.partial_cmp(&r.0).unwrap())
            .into_iter()
            .take(self.conf.population_size)
            .map(|x| x.1)
            .collect()
    }

    fn select_one(&self, population: Vec<Mutator>) -> Mutator {
        population.into_iter()
            .map(|x| (x.as_network_buf().as_network().error(self.conf.train_conf.error_conf), x))
            .sorted_by(|l, r| l.0.partial_cmp(&r.0).unwrap())
            .into_iter()
            .nth(0)
            .unwrap().1
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
        group_size: 1000,
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
        0.0, 0.0, 0.0, 0.24203979001903472, 0.0,
        0.0, 0.0, 0.0, 0.4806181555400138, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.5457742441788879,
        0.028733663238170292, 0.02873366324175895, 0.02588608755999839, 0.028733663241916427, 0.028733663241779898,
        0.028733663237706847, 0.0013892027510078162, 0.028733663238651758, 0.0013892027510077972, 0.1764921297127303,
    ] as &[f64]);
}
