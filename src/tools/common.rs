extern crate rustc_serialize;

use std::io::{BufRead, Lines};
use self::rustc_serialize::{Decodable, json};

use super::super::neural_network::NetworkBuf;

#[derive(RustcDecodable)]
pub struct Sample {
    pub input: Vec<f64>,
    pub output: Vec<f64>,
}

pub fn make_conf<T: Decodable>(file_path: &str) -> T {
    use std::io::Read;
    use std::fs::File;
    let mut data = String::new();
    File::open(file_path).unwrap().read_to_string(&mut data).unwrap();
    json::decode(&data).unwrap()
}

pub fn make_network_buf(file_path: &str) -> NetworkBuf {
    use std::io::Read;
    use std::fs::File;
    let mut data = String::new();
    File::open(file_path).unwrap().read_to_string(&mut data).unwrap();
    json::decode(&data).unwrap()
}

pub fn make_samples(file_path: &str) -> Vec<Sample> {
    use std::io::{BufReader, stdin};
    use std::fs::File;
    if file_path.is_empty() {
        let file = stdin();
        return make_samples_data(file.lock().lines());
    } else {
        make_samples_data(BufReader::new(File::open(file_path).unwrap()).lines())
    }
}

fn make_samples_data<B: BufRead>(lines: Lines<B>) -> Vec<Sample> {
    lines.map(|line| json::decode::<Sample>(&line.unwrap()).unwrap()).collect::<_>()
}
