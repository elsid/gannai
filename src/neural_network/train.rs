extern crate bobyqa;

use self::bobyqa::Bobyqa;

use super::common::Value;
use super::error::Error;
use super::error;
use super::matrix::Matrix;
use super::network::{Network, NetworkMut};

pub struct Conf<'r> {
    pub error_conf: &'r error::Conf<'r>,
}

pub trait Train {
    fn train<'r>(&mut self, conf: &Conf) -> Value;
}

impl<'network> Train for NetworkMut<'network> {
    fn train<'r>(&mut self, conf: &Conf) -> Value {
        use std::iter::repeat;
        let nodes_count = self.weights.column_len();
        let inputs = &self.inputs;
        let outputs = &self.outputs;
        let nodes = &self.nodes;
        let error_function = |weights_values: &[Value]| {
            assert!(weights_values.len() >= nodes_count * nodes_count);
            let weights = Matrix::new(nodes_count, weights_values);
            let network = Network {inputs: inputs, outputs: outputs,
                                   weights: weights, nodes: nodes};
            network.error(conf.error_conf)
        };
        let variables_count = self.weights.values().len();
        let lower_bound = repeat(0.0).take(variables_count).collect::<Vec<_>>();
        let upper_bound = repeat(1.0).take(variables_count).collect::<Vec<_>>();
        Bobyqa::new()
            .variables_count(variables_count)
//            .number_of_interpolation_conditions((variables_count + 1)*(variables_count + 2)/2)
            .number_of_interpolation_conditions(variables_count + 2)
            .lower_bound(&lower_bound[..])
            .upper_bound(&upper_bound[..])
            .perform(self.weights.values(), &error_function)
    }
}
