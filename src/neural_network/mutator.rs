use std::collections::btree_map;
use std::collections::btree_set;
use std::collections::{BTreeSet, HashSet};

use super::common::{Node, Weight};
use super::graph::Graph;
use super::id_generator::IdGenerator;
use super::network::{Connection, Network, NetworkBuf};

pub use super::graph::Arc;

#[derive(Clone, Debug)]
pub struct Mutator {
    inputs: BTreeSet<Node>,
    outputs: HashSet<Node>,
    graph: Graph,
}

impl Mutator {
    pub fn new(node_id: &mut IdGenerator, inputs_count: usize,
               outputs_count: usize, weight: Weight) -> Self {
        let inputs = (0..inputs_count)
            .map(|_| Node(node_id.generate()))
            .collect::<BTreeSet<_>>();
        let outputs = (0..outputs_count)
            .map(|_| Node(node_id.generate()))
            .collect::<HashSet<_>>();
        let mut graph = Graph::new();
        for input in inputs.iter() {
            graph.add_node(*input);
            for output in outputs.iter() {
                graph.add_node(*output);
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

    pub fn nodes(&self) -> btree_set::Iter<Node> {
        self.graph.nodes().iter()
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
        let middle = Node(node_id.generate());
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
                graph.add_arc(src_node, network.nodes[&dst], weight);
            }
        }
        Mutator {
            inputs: network.inputs.iter().map(|x| network.nodes[x]).collect::<_>(),
            outputs: network.outputs.iter().map(|x| network.nodes[x]).collect::<_>(),
            graph: graph,
        }
    }

    pub fn union(&self, other: &Mutator) -> Self {
        Mutator {
            inputs: self.inputs.union(&other.inputs).cloned().collect(),
            outputs: self.outputs.union(&other.outputs).cloned().collect(),
            graph: self.graph.union(&other.graph),
        }
    }
}

#[test]
fn test_new_should_succeed() {
    let mut node_id = IdGenerator::new(0);
    Mutator::new(&mut node_id, 4, 3, 0.1);
}
