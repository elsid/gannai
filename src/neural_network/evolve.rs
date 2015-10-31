extern crate itertools;
extern crate rand;

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
        let population = (0..self.conf.population_size)
            .map(|_| self.mutator.clone()).collect::<Vec<_>>();
        let evolved = self.evolve(population);
        self.select_one(evolved)
    }

    fn evolve(&mut self, population: Vec<Mutator>) -> Vec<Mutator> {
        let mut mutated = self.mutate(population);
        while !self.terminate(&mutated) {
            let propagated = self.propagate(mutated);
            let selected = self.select(propagated);
            mutated = self.mutate(selected);
            self.iterations_count += 1;
        }
        mutated
    }

    fn mutate(&mut self, population: Vec<Mutator>) -> Vec<Mutator> {
        let ref mut node_id = self.conf.node_id;
        let ref mut rng = self.conf.rng;
        let ref train_conf = self.conf.train_conf;
        population.into_iter()
            .map(|mut x| {
                for _ in 0..3 {
                    match rng.gen_range(0, 2) {
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
            .map(|x| {
                let mut network_buf = x.as_network_buf();
                network_buf.as_network_mut().train(train_conf);
                Mutator::from_network(&network_buf.as_network())
            })
            .collect()
    }

    fn terminate(&self, population: &Vec<Mutator>) -> bool {
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
