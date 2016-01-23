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

#[test]
fn test_train_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use neural_network::common::Node;
    use neural_network::matrix::MatrixMut;
    use neural_network::apply::{Conf as ApplyConf};
    use neural_network::error::{Conf as ErrorConf, Sample};
    let mut weights_values = [
        0.1, 0.1,
        0.1, 0.1,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [1].iter().cloned().collect::<HashSet<usize>>();
    {
        let weights = MatrixMut::new(2, &mut weights_values);
        let nodes = (0..2).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
        let mut network = NetworkMut {
            inputs: &inputs,
            outputs: &outputs,
            weights: weights,
            nodes: &nodes
        };
        let apply_conf = ApplyConf {
            group_size: 1000,
            threshold: 1e-4,
        };
        let samples = [
            Sample {input: &[0.5], output: &[0.4]},
        ];
        let error_conf = ErrorConf {
            apply_conf: &apply_conf,
            samples: &samples,
        };
        let conf = Conf {
            error_conf: &error_conf,
        };
        let error = network.train(&conf);
        assert_eq!(error, 0.0000000024013835364655733);
    }
    assert_eq!(weights_values, [
        0.7500085532432279, 0.9999933043615582,
        0.986926343197382, 0.9995644922815343,
    ]);
}
