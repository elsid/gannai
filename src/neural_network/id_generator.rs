#[derive(Clone, Debug)]
pub struct IdGenerator {
    next_id: usize,
}

impl IdGenerator {
    pub fn new(initial: usize) -> IdGenerator {
        IdGenerator {next_id: initial}
    }

    pub fn generate(&mut self) -> usize {
        let current = self.next_id;
        self.next_id = current + 1;
        current
    }
}

#[test]
fn test_new_should_succeed() {
    IdGenerator::new(42);
}

#[test]
fn test_first_generated_should_be_equal_to_initial() {
    assert_eq!(IdGenerator::new(42).generate(), 42);
}
