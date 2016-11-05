extern crate dot;

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};

use super::common::{Node, Weight};

#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct Arc(pub Node, pub Node);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeArcs {
    pub outgoing: HashSet<Arc>,
    pub incoming: HashSet<Arc>,
}

pub type Nodes = BTreeMap<Node, NodeArcs>;
pub type Arcs = BTreeMap<Arc, Weight>;

#[derive(Clone)]
pub enum ArcsType {
    Outgoing,
    Incoming,
}

impl ArcsType {
    pub fn arcs<'r>(&self, node_arcs: &'r NodeArcs) -> &'r HashSet<Arc> {
        match *self {
            ArcsType::Outgoing => &node_arcs.outgoing,
            ArcsType::Incoming => &node_arcs.incoming,
        }
    }

    pub fn neighborhood(&self, arc: &Arc) -> Node {
        let &Arc(src, dst) = arc;
        match *self {
            ArcsType::Outgoing => dst,
            ArcsType::Incoming => src,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
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

    pub fn node_arcs(&self, node: &Node) -> &NodeArcs {
        &self.nodes[node]
    }

    pub fn arc_weight(&self, arc: &Arc) -> Weight {
        self.arcs[arc]
    }

    pub fn add_node(&mut self, id: Node) {
        assert!(!self.nodes.contains_key(&id));
        self.nodes.insert(id, NodeArcs {outgoing: HashSet::new(), incoming: HashSet::new()});
    }

    pub fn rm_node(&mut self, id: &Node) {
        {
            let arcs = self.nodes.get(id).unwrap();
            assert!(arcs.incoming.is_empty() && arcs.outgoing.is_empty());
        }
        self.nodes.remove(id);
    }

    pub fn add_arc(&mut self, src: Node, dst: Node, weight: Weight) -> Arc {
        assert!(self.nodes.contains_key(&src));
        assert!(self.nodes.contains_key(&dst));
        let arc = Arc(src, dst);
        assert!(!self.arcs.contains_key(&arc));
        self.arcs.insert(arc.clone(), weight);
        self.nodes.get_mut(&src).unwrap().outgoing.insert(arc.clone());
        self.nodes.get_mut(&dst).unwrap().incoming.insert(arc.clone());
        arc
    }

    pub fn rm_arc(&mut self, arc: &Arc) {
        self.arcs.remove(arc);
        let &Arc(src, dst) = arc;
        self.nodes.get_mut(&src).unwrap().outgoing.remove(arc);
        self.nodes.get_mut(&dst).unwrap().incoming.remove(arc);
    }

    pub fn union(&self, other: &Graph) -> Graph {
        let mut nodes = self.nodes.keys().cloned().collect::<HashSet<_>>()
            .union(&other.nodes.keys().cloned().collect::<HashSet<_>>())
            .map(|x| (*x, NodeArcs {outgoing: HashSet::new(), incoming: HashSet::new()}))
            .collect::<Nodes>();
        let self_arcs = self.arcs.keys().cloned().collect::<HashSet<_>>();
        let other_arcs = other.arcs.keys().cloned().collect::<HashSet<_>>();
        let arcs: Arcs = self_arcs.union(&other_arcs).cloned().into_iter()
            .map(|arc| {
                let self_w = self.arcs.get(&arc);
                let other_w = other.arcs.get(&arc);
                let n = self_w.is_some() as i64 + other_w.is_some() as i64;
                let weight = (self_w.unwrap_or(&0.0) + other_w.unwrap_or(&0.0)) / n as Weight;
                (arc, weight)
            }).collect();
        for arc in arcs.keys() {
            let &Arc(src, dst) = arc;
            nodes.get_mut(&src).unwrap().outgoing.insert(arc.clone());
            nodes.get_mut(&dst).unwrap().incoming.insert(arc.clone());
        }
        Graph {nodes: nodes, arcs: arcs}
    }

    pub fn unreachable_from<'r, Nodes>(&self, nodes: Nodes) -> HashSet<Node>
            where Nodes: Iterator<Item=&'r Node> {
        self.unreachable(nodes, ArcsType::Outgoing)
    }

    pub fn unreachable_to<'r, Nodes>(&self, nodes: Nodes) -> HashSet<Node>
            where Nodes: Iterator<Item=&'r Node> {
        self.unreachable(nodes, ArcsType::Incoming)
    }

    pub fn unreachable<'r, Nodes>(&self, nodes: Nodes, arcs_type: ArcsType) -> HashSet<Node>
            where Nodes: Iterator<Item=&'r Node> {
        let mut visited = HashSet::new();
        for node in nodes {
            self.connected_component(node, arcs_type.clone(), &mut visited);
        }
        self.nodes.keys().cloned().collect::<HashSet<_>>().difference(&visited).cloned().collect()
    }

    pub fn connected_component(&self, initial: &Node, arcs_type: ArcsType, visited: &mut HashSet<Node>) {
        assert!(self.nodes.contains_key(&initial));
        if visited.contains(initial) {
            return;
        }
        let mut nodes = vec![initial.clone()];
        while !nodes.is_empty() {
            let node = nodes.pop().unwrap();
            visited.insert(node);
            for arc in arcs_type.arcs(self.nodes.get(&node).unwrap()).iter() {
                let node = arcs_type.neighborhood(arc);
                if !visited.contains(&node) {
                    nodes.push(node);
                }
            }
        }
    }
}

impl<'a> dot::Labeller<'a, Node, Arc> for Graph {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("neural_network").unwrap()
    }

    fn node_id(&'a self, node: &Node) -> dot::Id<'a> {
        let &Node(id) = node;
        dot::Id::new(format!("node_{}", id)).unwrap()
    }

    fn node_label<'b>(&'b self, node: &Node) -> dot::LabelText<'b> {
        let &Node(id) = node;
        dot::LabelText::LabelStr(Cow::Owned(format!("{}", id)))
    }

    fn edge_label<'b>(&'b self, arc: &Arc) -> dot::LabelText<'b> {
        let weight = self.arc_weight(arc);
        dot::LabelText::LabelStr(Cow::Owned(format!("{}", weight)))
    }
}

impl<'a> dot::GraphWalk<'a, Node, Arc> for Graph {
    fn nodes(&self) -> dot::Nodes<'a, Node> {
        Cow::Owned(self.nodes().keys().cloned().collect::<Vec<Node>>())
    }

    fn edges(&'a self) -> dot::Edges<'a, Arc> {
        Cow::Owned(self.arcs().keys().cloned().collect::<Vec<Arc>>())
    }

    fn source(&self, arc: &Arc) -> Node {
        let &Arc(src, _) = arc;
        src
    }

    fn target(&self, arc: &Arc) -> Node {
        let &Arc(_, dst) = arc;
        dst
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
    assert!(graph.nodes().contains_key(&Node(42)));
}

#[test]
fn test_rm_node_without_arcs_should_succeed() {
    let mut graph = Graph::new();
    graph.add_node(Node(42));
    graph.rm_node(&Node(42));
    assert!(graph.nodes().is_empty());
}

#[test]
#[should_panic]
fn test_rm_node_with_arcs_should_panic() {
    let mut graph = Graph::new();
    let src = Node(1);
    let dst = Node(2);
    graph.add_node(src);
    graph.add_node(dst);
    graph.add_arc(src, dst, 1.0);
    graph.rm_node(&src);
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
    assert_eq!(graph.node_arcs(&src), &NodeArcs {
        outgoing: [arc.clone()].iter().cloned().collect(),
        incoming: HashSet::new(),
    });
    assert_eq!(graph.node_arcs(&dst), &NodeArcs {
        outgoing: HashSet::new(),
        incoming: [arc.clone()].iter().cloned().collect(),
    });
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
    let node1_arcs = NodeArcs {
        outgoing: [Arc(node1, node2), Arc(node1, node4)].iter().cloned().collect(),
        incoming: HashSet::new(),
    };
    let node2_arcs = NodeArcs {
        outgoing: [Arc(node2, node3)].iter().cloned().collect(),
        incoming: [Arc(node1, node2)].iter().cloned().collect(),
    };
    let node3_arcs = NodeArcs {
        outgoing: HashSet::new(),
        incoming: [Arc(node2, node3)].iter().cloned().collect(),
    };
    let node4_arcs = NodeArcs {
        outgoing: HashSet::new(),
        incoming: [Arc(node1, node4)].iter().cloned().collect(),
    };
    assert_eq!(union.nodes(), &[
        (node1, node1_arcs),
        (node2, node2_arcs),
        (node3, node3_arcs),
        (node4, node4_arcs),
    ].iter().cloned().collect::<Nodes>());
    assert_eq!(union.arcs(), &[
        (Arc(node1, node2), 0.75),
        (Arc(node2, node3), 1.0),
        (Arc(node1, node4), 1.0),
    ].iter().cloned().collect::<Arcs>());
}

#[test]
fn test_connected_component_by_outgoing_for_src_with_arc_to_dst_should_return_both() {
    let mut graph = Graph::new();
    let src = Node(1);
    let dst = Node(2);
    graph.add_node(src);
    graph.add_node(dst);
    graph.add_arc(src, dst, 1.0);
    let mut visited = HashSet::new();
    graph.connected_component(&src, ArcsType::Outgoing, &mut visited);
    assert_eq!(visited, [src, dst].iter().cloned().collect());
}

#[test]
fn test_connected_component_by_outgoing_for_dst_with_arc_from_src_return_dst() {
    let mut graph = Graph::new();
    let src = Node(1);
    let dst = Node(2);
    graph.add_node(src);
    graph.add_node(dst);
    graph.add_arc(src, dst, 1.0);
    let mut visited = HashSet::new();
    graph.connected_component(&dst, ArcsType::Outgoing, &mut visited);
    assert_eq!(visited, [dst].iter().cloned().collect());
}

#[test]
fn test_connected_component_by_incoming_for_src_with_arc_to_dst_should_return_src() {
    let mut graph = Graph::new();
    let src = Node(1);
    let dst = Node(2);
    graph.add_node(src);
    graph.add_node(dst);
    graph.add_arc(src, dst, 1.0);
    let mut visited = HashSet::new();
    graph.connected_component(&src, ArcsType::Incoming, &mut visited);
    assert_eq!(visited, [src].iter().cloned().collect());
}

#[test]
fn test_connected_component_by_incoming_for_dst_with_arc_from_src_return_both() {
    let mut graph = Graph::new();
    let src = Node(1);
    let dst = Node(2);
    graph.add_node(src);
    graph.add_node(dst);
    graph.add_arc(src, dst, 1.0);
    let mut visited = HashSet::new();
    graph.connected_component(&dst, ArcsType::Incoming, &mut visited);
    assert_eq!(visited, [src, dst].iter().cloned().collect());
}

#[test]
fn test_unreachable_from_for_two_return_nothing() {
    let mut graph = Graph::new();
    let first = Node(1);
    let second = Node(2);
    graph.add_node(first);
    graph.add_node(second);
    let result = graph.unreachable_from([first, second].iter());
    assert_eq!(result, HashSet::new());
}

#[test]
fn test_unreachable_from_for_first_where_first_connected_to_second_should_return_nothing() {
    let mut graph = Graph::new();
    let first = Node(1);
    let second = Node(2);
    graph.add_node(first);
    graph.add_node(second);
    graph.add_arc(first, second, 1.0);
    let result = graph.unreachable_from([first].iter());
    assert_eq!(result, HashSet::new());
}

#[test]
fn test_unreachable_from_for_first_and_second_where_first_connected_to_second_should_return_nothing() {
    let mut graph = Graph::new();
    let first = Node(1);
    let second = Node(2);
    graph.add_node(first);
    graph.add_node(second);
    graph.add_arc(first, second, 1.0);
    let result = graph.unreachable_from([first, second].iter());
    assert_eq!(result, HashSet::new());
}

#[test]
fn test_unreachable_from_for_first_where_second_connected_to_first_should_return_second() {
    let mut graph = Graph::new();
    let first = Node(1);
    let second = Node(2);
    graph.add_node(first);
    graph.add_node(second);
    graph.add_arc(second, first, 1.0);
    let result = graph.unreachable_from([first].iter());
    assert_eq!(result, [second].iter().cloned().collect());
}

#[test]
fn test_unreachable_to_for_two_return_nothing() {
    let mut graph = Graph::new();
    let first = Node(1);
    let second = Node(2);
    graph.add_node(first);
    graph.add_node(second);
    let result = graph.unreachable_to([first, second].iter());
    assert_eq!(result, HashSet::new());
}

#[test]
fn test_unreachable_to_for_first_where_first_connected_to_second_should_return_second() {
    let mut graph = Graph::new();
    let first = Node(1);
    let second = Node(2);
    graph.add_node(first);
    graph.add_node(second);
    graph.add_arc(first, second, 1.0);
    let result = graph.unreachable_to([first].iter());
    assert_eq!(result, [second].iter().cloned().collect());
}

#[test]
fn test_unreachable_to_for_first_where_second_connected_to_first_should_return_nothing() {
    let mut graph = Graph::new();
    let first = Node(1);
    let second = Node(2);
    graph.add_node(first);
    graph.add_node(second);
    graph.add_arc(second, first, 1.0);
    let result = graph.unreachable_to([first].iter());
    assert_eq!(result, HashSet::new());
}

#[test]
fn test_unreachable_to_for_first_and_second_where_second_connected_to_first_should_return_nothing() {
    let mut graph = Graph::new();
    let first = Node(1);
    let second = Node(2);
    graph.add_node(first);
    graph.add_node(second);
    graph.add_arc(second, first, 1.0);
    let result = graph.unreachable_to([first, second].iter());
    assert_eq!(result, HashSet::new());
}
