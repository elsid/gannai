use std::collections::{BTreeMap, HashSet};

use super::common::{Node, Weight};

#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct Arc(pub Node, pub Node);

pub type Nodes = HashSet<Node>;
type Arcs = BTreeMap<Arc, Weight>;

#[derive(Clone, Debug)]
pub struct Graph {
    nodes: Nodes,
    arcs: Arcs,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            nodes: Nodes::new(),
            arcs: Arcs::new(),
        }
    }

    pub fn nodes(&self) -> &Nodes {
        &self.nodes
    }

    pub fn arcs(&self) -> &Arcs {
        &self.arcs
    }

    pub fn add_node(&mut self, id: Node) {
        self.nodes.insert(id);
    }

    pub fn rm_node(&mut self, id: Node) {
        self.nodes.remove(&id);
    }

    pub fn add_arc(&mut self, src: Node, dst: Node, weight: Weight) -> Arc {
        assert!(self.nodes.contains(&src));
        assert!(self.nodes.contains(&dst));
        let arc = Arc(src, dst);
        self.arcs.insert(arc.clone(), weight);
        arc
    }

    pub fn rm_arc(&mut self, arc: &Arc) {
        self.arcs.remove(arc);
    }

    pub fn union(&self, other: &Graph) -> Graph {
        let nodes = self.nodes.union(&other.nodes).cloned().collect();
        let self_arcs = self.arcs.keys().cloned().collect::<HashSet<_>>();
        let other_arcs = other.arcs.keys().cloned().collect::<HashSet<_>>();
        let arcs = self_arcs.union(&other_arcs).cloned().into_iter()
            .map(|arc| {
                let self_w = self.arcs.get(&arc);
                let other_w = other.arcs.get(&arc);
                let n = self_w.is_some() as i64 + other_w.is_some() as i64;
                let weight = (self_w.unwrap_or(&0.0) + other_w.unwrap_or(&0.0)) / n as Weight;
                (arc, weight)
            }).collect();
        Graph {nodes: nodes, arcs: arcs}
    }
}

#[test]
fn test_create_should_succeed() {
    Graph::new();
}

#[test]
fn test_add_node_should_succeed() {
    let mut graph = Graph::new();
    graph.add_node(Node(42));
    assert!(graph.nodes().contains(&Node(42)));
}

#[test]
fn test_rm_node_should_succeed() {
    let mut graph = Graph::new();
    graph.add_node(Node(42));
    graph.rm_node(Node(42));
    assert!(graph.nodes().is_empty());
}

#[test]
fn test_add_arc_should_succeed() {
    let mut graph = Graph::new();
    let src = Node(1);
    let dst = Node(2);
    let weight = 0.3;
    graph.add_node(src);
    graph.add_node(dst);
    let arc = graph.add_arc(src, dst, weight);
    assert_eq!(arc, Arc(src, dst));
    assert_eq!(graph.arcs()[&arc], weight);
}

#[test]
fn test_rm_arc_should_succeed() {
    let mut graph = Graph::new();
    let src = Node(1);
    let dst = Node(2);
    let weight = 0.3;
    graph.add_node(src);
    graph.add_node(dst);
    let arc = graph.add_arc(src, dst, weight);
    graph.rm_arc(&arc);
    assert!(graph.arcs().is_empty());
}

#[test]
#[should_panic]
fn test_add_arc_with_nonexistent_nodes_should_panic() {
    Graph::new().add_arc(Node(1), Node(2), 0.3);
}

#[test]
fn test_union_should_succeed() {
    let node1 = Node(1);
    let node2 = Node(2);
    let node3 = Node(3);
    let node4 = Node(4);
    let mut graph1 = Graph::new();
    graph1.add_node(node1);
    graph1.add_node(node2);
    graph1.add_node(node3);
    graph1.add_arc(node1, node2, 1.0);
    graph1.add_arc(node2, node3, 1.0);
    let mut graph2 = Graph::new();
    graph2.add_node(node1);
    graph2.add_node(node2);
    graph2.add_node(node4);
    graph2.add_arc(node1, node2, 0.5);
    graph2.add_arc(node1, node4, 1.0);
    let union = graph1.union(&graph2);
    assert_eq!(union.nodes(), &[node1, node2, node3, node4].iter().cloned().collect::<Nodes>());
    assert_eq!(union.arcs(), &[
        (Arc(node1, node2), 0.75),
        (Arc(node2, node3), 1.0),
        (Arc(node1, node4), 1.0),
    ].iter().cloned().collect::<Arcs>());
}
