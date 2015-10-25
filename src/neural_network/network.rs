use std::collections::{BTreeSet, HashMap, HashSet};

use super::common::{Node, Weight};
use super::matrix::{Matrix, MatrixMut, MatrixBuf};

#[derive(Debug)]
pub struct Network<'r> {
    pub inputs: &'r BTreeSet<usize>,
    pub outputs: &'r HashSet<usize>,
    pub weights: Matrix<'r, Weight>,
    pub nodes: &'r HashMap<usize, Node>,
}

#[derive(Debug)]
pub struct NetworkMut<'r> {
    pub inputs: &'r BTreeSet<usize>,
    pub outputs: &'r HashSet<usize>,
    pub weights: MatrixMut<'r, Weight>,
    pub nodes: &'r HashMap<usize, Node>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Connection {
    pub src: Node,
    pub dst: Node,
    pub weight: Weight,
}

#[derive(Debug, Clone)]
pub struct NetworkBuf {
    inputs: BTreeSet<usize>,
    outputs: HashSet<usize>,
    weights: MatrixBuf<Weight>,
    nodes: HashMap<usize, Node>,
}

impl NetworkBuf {
    pub fn new<'r, Connections, OrdNodeIds, NodeIds>(
            arcs: Connections,
            inputs: OrdNodeIds,
            outputs: NodeIds) -> NetworkBuf
            where Connections: Iterator<Item=Connection> + Clone,
                  OrdNodeIds: Iterator<Item=&'r Node> + Clone,
                  NodeIds: Iterator<Item=&'r Node> + Clone {
        let mut indicies = HashMap::new();
        for arc in arcs.clone() {
            if !indicies.contains_key(&arc.src) {
                let index = indicies.len();
                indicies.insert(arc.src, index);
            }
            if !indicies.contains_key(&arc.dst) {
                let index = indicies.len();
                indicies.insert(arc.dst, index);
            }
        }
        let mut weights = MatrixBuf::new(indicies.len(), 0.0);
        {
            let mut w = weights.as_matrix_mut();
            for arc in arcs {
                w.set(indicies[&arc.src], indicies[&arc.dst], arc.weight);
            }
        }
        NetworkBuf {
            inputs: inputs.map(|x| indicies[x]).collect::<_>(),
            outputs: outputs.map(|x| indicies[x]).collect::<_>(),
            weights: weights,
            nodes: indicies.iter().map(|(&k, &v)| (v, k)).collect::<_>(),
        }
    }

    pub fn as_network<'r>(&'r self) -> Network<'r> {
        Network {
            inputs: &self.inputs,
            outputs: &self.outputs,
            weights: self.weights.as_matrix(),
            nodes: &self.nodes,
        }
    }

    pub fn as_network_mut<'r>(&'r mut self) -> NetworkMut<'r> {
        NetworkMut {
            inputs: &self.inputs,
            outputs: &self.outputs,
            weights: self.weights.as_matrix_mut(),
            nodes: &self.nodes,
        }
    }
}
