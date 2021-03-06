pub type Value = f64;
pub type Weight = Value;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, RustcDecodable, RustcEncodable)]
pub struct Node(pub usize);
