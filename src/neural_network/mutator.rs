use std::collections::btree_map;
use std::collections::BTreeSet;

use super::common::{Node, Weight};
use super::graph::{Graph, NodeArcs};
use super::id_generator::IdGenerator;
use super::network::{Connection, Network, NetworkBuf};

pub use super::graph::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct Mutator {
    inputs: BTreeSet<Node>,
    outputs: BTreeSet<Node>,
    graph: Graph,
}

impl Mutator {
    pub fn new(node_id: &mut IdGenerator, inputs_count: usize,
               outputs_count: usize, weight: Weight) -> Self {
        assert!(weight > 0.0);
        let inputs = (0..inputs_count)
            .map(|_| Node(node_id.generate()))
            .collect::<BTreeSet<_>>();
        let outputs = (0..outputs_count)
            .map(|_| Node(node_id.generate()))
            .collect::<BTreeSet<_>>();
        let mut graph = Graph::new();
        for input in inputs.iter() {
            if !graph.nodes().contains_key(input) {
                graph.add_node(*input);
            }
            for output in outputs.iter() {
                if !graph.nodes().contains_key(output) {
                    graph.add_node(*output);
                }
                graph.add_arc(*input, *output, weight);
            }
        }
        Mutator {
            inputs: inputs,
            outputs: outputs,
            graph: graph,
        }
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn nodes(&self) -> btree_map::Keys<Node, NodeArcs> {
        self.graph.nodes().keys()
    }

    pub fn arcs(&self) -> btree_map::Keys<Arc, Weight> {
        self.graph.arcs().keys()
    }

    pub fn add_arc(&mut self, src: Node, dst: Node, weight: Weight) -> &mut Self {
        self.graph.add_arc(src, dst, weight);
        self
    }

    pub fn rm_arc(&mut self, arc: &Arc) -> &mut Self {
        self.graph.rm_arc(arc);
        self
    }

    pub fn split(&mut self, node_id: &mut IdGenerator, arc: &Arc) -> &mut Self {
        let mut middle = Node(node_id.generate());
        while self.graph.nodes().contains_key(&middle) {
            middle = Node(node_id.generate());
        }
        self.graph.add_node(middle);
        let &Arc(src, dst) = arc;
        let weight = self.graph.arcs()[arc].sqrt();
        self.graph.add_arc(src, middle, weight);
        self.graph.add_arc(middle, dst, weight);
        self.graph.rm_arc(arc);
        self
    }

    pub fn as_network_buf(&self) -> NetworkBuf {
        let as_connection = |(&Arc(src, dst), &weight)| {
            Connection {src: src, dst: dst, weight: weight}
        };
        let arcs = self.graph.arcs().iter().map(&as_connection);
        NetworkBuf::new(arcs, self.inputs.iter(), self.outputs.iter())
    }

    pub fn from_network(network: &Network) -> Mutator {
        let mut graph = Graph::new();
        for (_, &node) in network.nodes.iter() {
            graph.add_node(node);
        }
        for (&src, &src_node) in network.nodes.iter() {
            for (dst, &weight) in network.weights.row(src).iter().enumerate() {
                if weight > 0.0 {
                    graph.add_arc(src_node, network.nodes[&dst], weight);
                }
            }
        }
        let inputs = network.inputs.iter().map(|v| network.nodes[v]).collect::<BTreeSet<_>>();
        let outputs = network.outputs.iter().map(|v| network.nodes[v]).collect::<BTreeSet<_>>();
        Mutator {inputs: inputs, outputs: outputs, graph: graph}
    }

    pub fn union(&self, other: &Mutator) -> Self {
        Mutator {
            inputs: self.inputs.union(&other.inputs).cloned().collect(),
            outputs: self.outputs.union(&other.outputs).cloned().collect(),
            graph: self.graph.union(&other.graph),
        }
    }

    pub fn rm_useless(&mut self) -> &mut Self {
        use std::collections::HashSet;
        let nodes = self.graph.unreachable_from(self.inputs.iter())
            .union(&self.graph.unreachable_to(self.outputs.iter()))
            .cloned().collect::<HashSet<_>>()
            .difference(&self.inputs.union(&self.outputs).cloned().collect::<HashSet<_>>())
            .cloned().collect::<Vec<_>>();
        let arcs = nodes.iter()
            .flat_map(|x| {
                let node_arcs = self.graph.nodes().get(x).unwrap();
                node_arcs.outgoing.union(&node_arcs.incoming).cloned()
            })
            .collect::<HashSet<_>>();
        for arc in arcs.iter() {
            self.graph.rm_arc(arc);
        }
        for node in nodes.iter() {
            self.graph.rm_node(node);
        }
        self
    }
}

#[test]
fn test_new_should_succeed() {
    let mut node_id = IdGenerator::new(0);
    Mutator::new(&mut node_id, 4, 3, 0.1);
}

#[test]
fn test_rm_last_arc_for_not_input_or_output_should_rm_node() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use neural_network::matrix::Matrix;
    use neural_network::network::Network;
    let mut weights_values = [
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [2].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(3, &mut weights_values);
    let nodes = (0..3).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {
        inputs: &inputs,
        outputs: &outputs,
        weights: weights,
        nodes: &nodes
    };
    let mut mutator = Mutator::from_network(&network);
    mutator.rm_arc(&Arc(Node(0), Node(1)));
    let network_buf = mutator.as_network_buf();
    let result = network_buf.as_network();
    assert_eq!(result.weights.values(), [
        0.0, 0.0,
        0.0, 0.0,
    ]);
}

#[test]
fn test_rm_useless_for_node_not_input_and_not_output_without_incoming_arcs_should_rm_arc_and_node() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use neural_network::matrix::Matrix;
    use neural_network::network::Network;
    let mut weights_values = [
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [2].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(3, &mut weights_values);
    let nodes = (0..3).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {
        inputs: &inputs,
        outputs: &outputs,
        weights: weights,
        nodes: &nodes
    };
    let mut mutator = Mutator::from_network(&network);
    mutator.rm_useless();
    let network_buf = mutator.as_network_buf();
    let result = network_buf.as_network();
    assert_eq!(result.weights.values(), [
        0.0, 0.0,
        0.0, 0.0,
    ]);
}

#[test]
fn test_rm_useless_for_not_connected_only_input_and_output_nodes_should_do_nothing() {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use neural_network::matrix::Matrix;
    use neural_network::network::Network;
    let mut weights_values = [
        0.0, 0.0,
        0.0, 0.0,
    ];
    let inputs = [0].iter().cloned().collect::<BTreeSet<usize>>();
    let outputs = [1].iter().cloned().collect::<HashSet<usize>>();
    let weights = Matrix::new(2, &mut weights_values);
    let nodes = (0..2).map(|x| (x, Node(x))).collect::<HashMap<usize, Node>>();
    let network = Network {
        inputs: &inputs,
        outputs: &outputs,
        weights: weights,
        nodes: &nodes
    };
    let mut mutator = Mutator::from_network(&network);
    let expected = mutator.clone();
    mutator.rm_useless();
    assert_eq!(expected, mutator);
}
