use super::apply::Apply;
use super::apply;
use super::common::Value;
use super::network::Network;

pub struct Conf<'r> {
    pub apply_conf: &'r apply::Conf,
    pub samples: &'r [Sample<'r>],
}

pub trait Error {
    fn error<'r>(&self, conf: &Conf<'r>) -> Value;
}

impl<'network> Error for Network<'network> {
    fn error<'r>(&self, conf: &Conf<'r>) -> Value {
        conf.samples.iter()
            .inspect(|&sample| {
                assert!(sample.input.len() >= self.inputs.len());
                assert!(sample.output.len() >= self.outputs.len());
            })
            .map(|&Sample{input, output: expected}| {
                self.apply(conf.apply_conf)
                    .perform(input)
                    .iter()
                    .zip(expected.iter())
                    .map(|(lhs, rhs)| (lhs - rhs).powi(2))
                    .fold(0.0, |sum, x| sum + x)
                    .sqrt()
            })
            .fold(0.0, |sum, x| sum + x)
    }
}

pub struct Sample<'r> {
    pub input: &'r [Value],
    pub output: &'r [Value],
}

#[test]
fn test_error_network_with_one_arc_and_one_sample_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use super::common::Node;
    use super::matrix::Matrix;
    let weight = 0.4;
    let weights_values = [
        0.0, weight,
        0.0, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [1].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(2, &weights_values);
    let nodes = (0..2).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {inputs: &inputs, outputs: &outputs, weights: weights, nodes: &nodes};
    let apply_conf = apply::Conf {group_len: 1, threshold: 1e-3};
    let input = 0.5;
    let output = 0.5;
    let samples = [
        Sample {input: &[input], output: &[output]},
    ];
    let conf = Conf {apply_conf: &apply_conf, samples: &samples};
    assert_eq!(network.error(&conf), (input * weight - output).powi(2).sqrt());
}

#[test]
fn test_error_network_with_one_arc_and_two_samples_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use super::common::Node;
    use super::matrix::Matrix;
    let w = 0.4;
    let weights_values = [
        0.0, w,
        0.0, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [1].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(2, &weights_values);
    let nodes = (0..2).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {inputs: &inputs, outputs: &outputs, weights: weights, nodes: &nodes};
    let apply_conf = apply::Conf {group_len: 1, threshold: 1e-3};
    let i1 = 0.4;
    let i2 = 0.6;
    let o1 = 0.5;
    let o2 = 0.7;
    let samples = [
        Sample {input: &[i1], output: &[o1]},
        Sample {input: &[i2], output: &[o2]},
    ];
    let conf = Conf {apply_conf: &apply_conf, samples: &samples};
    assert_eq!(network.error(&conf), (i1 * w - o1).powi(2).sqrt() + (i2 * w - o2).powi(2).sqrt());
}

#[test]
fn test_error_network_with_two_arcs_and_two_outputs_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use super::common::Node;
    use super::matrix::Matrix;
    let w12 = 0.4;
    let w13 = 0.2;
    let weights_values = [
        0.0, w12, w13,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [1, 2].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(3, &weights_values);
    let nodes = (0..3).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {inputs: &inputs, outputs: &outputs, weights: weights, nodes: &nodes};
    let apply_conf = apply::Conf {group_len: 1, threshold: 1e-3};
    let i = 0.4;
    let o1 = 0.4;
    let o2 = 0.6;
    let samples = [
        Sample {input: &[i], output: &[o1, o2]},
    ];
    let conf = Conf {apply_conf: &apply_conf, samples: &samples};
    assert_eq!(network.error(&conf),
               ((i * w12 / 2.0 - o1).powi(2) + (i * w13 / 2.0 - o2).powi(2)).sqrt());
}
