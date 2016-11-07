use std::collections::BTreeMap;

use super::common::Value;
use super::network::Network;

#[derive(RustcDecodable)]
pub struct Conf {
    pub threshold: Value,
}

pub trait Apply {
    fn apply<'r>(&'r self, conf: &'r Conf) -> Application<'r>;
}

impl<'network> Apply for Network<'network> {
    fn apply<'r>(&'r self, conf: &'r Conf) -> Application<'r> {
        Application::new(self, conf)
    }
}

struct ValuesGroup {
    pub sum: Value,
    pub node: usize,
}

pub struct Application<'r> {
    conf: &'r Conf,
    network: &'r Network<'r>,
}

impl<'r> Application<'r> {
    pub fn new(network: &'r Network<'r>, conf: &'r Conf) -> Self {
        Application {network: network, conf: conf}
    }

    pub fn perform(&self, values: &[Value]) -> Vec<Value> {
        assert!(values.len() >= self.network.inputs.len());
        let mut result: BTreeMap<usize, Value> = self.network.outputs.iter()
            .map(|&x| (x, 0.0)).collect::<_>();
        {
            let groups = self.network.inputs.iter().zip(values)
                .map(|(&node, &value)| {
                    self.perform_one(ValuesGroup {sum: value, node: node})
                })
                .flat_map(|x| x.into_iter());
            for group in groups {
                *result.get_mut(&group.node).unwrap() += group.sum;
            }
        }
        result.values().map(|&x| x).collect()
    }

    fn perform_one(&self, group: ValuesGroup) -> Vec<ValuesGroup> {
        if group.sum.abs() > self.conf.threshold {
            let mut result = self.network.weights.row(group.node).iter()
                .enumerate()
                .map(|(node, &weight)| {
                    self.perform_one(ValuesGroup {sum: group.sum * weight, node: node})
                })
                .flat_map(|x| x.into_iter())
                .collect::<Vec<_>>();
            if self.network.outputs.contains(&group.node) {
                result.push(ValuesGroup {sum: group.sum, node: group.node});
            }
            result
        } else {
            vec![]
        }
    }
}

#[test]
fn test_apply_network_contains_one_arc_with_positive_weight_should_succeed() {
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
    let conf = Conf {threshold: 1e-3};
    let input = 0.6;
    assert_eq!(&network.apply(&conf).perform(&[input])[..], &[input * weight]);
}

#[test]
fn test_apply_network_contains_one_arc_with_negative_weight_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use super::common::Node;
    use super::matrix::Matrix;
    let weight = -0.4;
    let weights_values = [
        0.0, weight,
        0.0, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [1].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(2, &weights_values);
    let nodes = (0..2).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {inputs: &inputs, outputs: &outputs, weights: weights, nodes: &nodes};
    let conf = Conf {threshold: 1e-3};
    let input = 0.6;
    assert_eq!(&network.apply(&conf).perform(&[input])[..], &[input * weight]);
}

#[test]
fn test_apply_network_with_two_arcs_and_two_inputs_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use super::common::Node;
    use super::matrix::Matrix;
    let w13 = 0.4;
    let w23 = 0.2;
    let weights_values = [
        0.0, 0.0, w13,
        0.0, 0.0, w23,
        0.0, 0.0, 0.0,
    ];
    let inputs = [0, 1].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [2].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(3, &weights_values);
    let nodes = (0..3).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {inputs: &inputs, outputs: &outputs, weights: weights, nodes: &nodes};
    let conf = Conf {threshold: 1e-3};
    let i1 = 0.6;
    let i2 = 0.7;
    assert_eq!(&network.apply(&conf).perform(&[i1, i2])[..], &[i1 * w13 + i2 * w23]);
}

#[test]
fn test_apply_network_with_two_arcs_and_two_outputs_should_succeed() {
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
    let conf = Conf {threshold: 1e-3};
    let input = 0.6;
    assert_eq!(&network.apply(&conf).perform(&[input])[..], &[input * w12, input * w13]);
}

#[test]
fn test_apply_network_with_two_arcs_and_one_middle_node_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use super::common::Node;
    use super::matrix::Matrix;
    let w12 = 0.4;
    let w23 = 0.2;
    let weights_values = [
        0.0, w12, 0.0,
        0.0, 0.0, w23,
        0.0, 0.0, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [2].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(3, &weights_values);
    let nodes = (0..3).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {inputs: &inputs, outputs: &outputs, weights: weights, nodes: &nodes};
    let conf = Conf {threshold: 1e-3};
    let input = 0.6;
    assert_eq!(&network.apply(&conf).perform(&[input])[..], &[input * w12 * w23]);
}

#[test]
fn test_apply_network_with_two_arcs_and_self_add_arced_input_node_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use super::common::Node;
    use super::matrix::Matrix;
    let w11 = 0.2;
    let w12 = 0.4;
    let weights_values = [
        w11, w12,
        0.0, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [1].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(2, &weights_values);
    let nodes = (0..2).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {inputs: &inputs, outputs: &outputs, weights: weights, nodes: &nodes};
    let conf = Conf {threshold: 1e-8};
    let input = 0.6;
    let actual = network.apply(&conf).perform(&[input]);
    let expected = [input * w12 / (1.0 - w11)];
    assert_eq!(actual.len(), expected.len());
    assert!((actual[0] - expected[0]).abs() <= 1e-8);
}

#[test]
fn test_apply_network_without_arcs_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use super::common::Node;
    use super::matrix::Matrix;
    let weights_values = [
        0.0, 0.0,
        0.0, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [1].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(2, &weights_values);
    let nodes = (0..2).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {inputs: &inputs, outputs: &outputs, weights: weights, nodes: &nodes};
    let conf = Conf {threshold: 1e-8};
    assert_eq!(&network.apply(&conf).perform(&[1.0]), &[0.0]);
}

#[test]
fn test_appy_network_with_two_arcs_cycle_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use super::common::Node;
    use super::matrix::Matrix;
    let w12 = 0.2;
    let w21 = 0.4;
    let weights_values = [
        0.0, w12,
        w21, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [1].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(2, &weights_values);
    let nodes = (0..2).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {inputs: &inputs, outputs: &outputs, weights: weights, nodes: &nodes};
    let conf = Conf {threshold: 1e-8};
    let input = 0.6;
    let actual = network.apply(&conf).perform(&[input]);
    let expected = [input * w12 / (1.0 - w12 * w21)];
    assert_eq!(actual.len(), expected.len());
    assert!((actual[0] - expected[0]).abs() <= 1e-8);
}

#[test]
fn test_apply_network_with_one_self_arced_node_should_succeed() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use super::common::Node;
    use super::matrix::Matrix;
    let weight = 0.2;
    let weights_values = [
        weight,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [0].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(1, &weights_values);
    let nodes = (0..1).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {inputs: &inputs, outputs: &outputs, weights: weights, nodes: &nodes};
    let conf = Conf {threshold: 1e-8};
    let input = 0.6;
    let actual = network.apply(&conf).perform(&[input]);
    let expected = [input / (1.0 - weight)];
    assert_eq!(actual.len(), expected.len());
    assert!((actual[0] - expected[0]).abs() <= 1e-8);
}
