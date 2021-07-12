use std::collections::HashMap;

pub struct Discriminator {
    number_of_hashtables: u16,
    h_rams: Vec<HashMap<u64, u16>>,
    times_trained: u64,
}

impl Discriminator {
    pub fn new(num: u16) -> Self {
        Self {
            number_of_hashtables: num,
            h_rams: Vec::new(),
            times_trained: 0,
        }
    }

    pub fn train(mut self, x: Vec<u64>) -> () {
        for i in 0..self.number_of_hashtables {
            let key = x[i as usize];
            let counter = self.h_rams[i as usize].entry(key).or_insert(0);
            *counter += 1;
        }
        self.times_trained += 1;
    }

    pub fn classify(self, x: Vec<u64>, bleach: u16) -> (u64, u64) {
        let mut votes: u64 = 0;
        for i in 0..self.number_of_hashtables {
            let key = x[i as usize];
            if let Some(x) = self.h_rams[i as usize].get(&key) {
                if *x > bleach {
                    votes += 1
                };
            }
        }
        (votes, self.times_trained)
    }
}
