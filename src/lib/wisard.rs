use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Discriminator {
    number_of_hashtables: u16,
    h_rams: Vec<HashMap<u64, u16>>,
    times_trained: u64,
}

impl Discriminator {
    pub fn new(num: u16) -> Discriminator {
        Discriminator {
            number_of_hashtables: num,
            h_rams: vec![HashMap::new(); num as usize],
            times_trained: 0,
        }
    }

    pub fn train(&mut self, x: Vec<u64>) {
        for i in 0..self.number_of_hashtables {
            let key = x[i as usize];
            let counter = self
                .h_rams
                .get_mut(i as usize)
                .unwrap()
                .entry(key)
                .or_insert(0);
            *counter += 1;
        }
        self.times_trained += 1;
    }

    pub fn classify(&self, x: &Vec<u64>, bleach: u16) -> (u64, u64) {
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

pub struct Wisard<T> {
    discs: HashMap<String, Discriminator>,
    addr_length: u64,
    number_of_hashtables: u16,
    mapping: Vec<u64>,
    last_rank: u64,
    rank_tables: HashMap<Vec<u64>, u64>,
    bleach: u16,
    phantom: PhantomData<T>,
}

impl<T> Wisard<T> {
    pub fn new(number_of_hashtables: u16, addr_length: u64, bleach: u16) -> Wisard<T>
    where
        T: PartialOrd + Copy,
    {
        // randomizes the mapping
        let mut rng_mapping =
            (0..addr_length as u64 * number_of_hashtables as u64).collect::<Vec<u64>>();
        rng_mapping.shuffle(&mut thread_rng());

        Wisard::<T> {
            discs: HashMap::new(),
            addr_length: addr_length,
            number_of_hashtables: number_of_hashtables,
            mapping: rng_mapping,
            last_rank: 0,
            rank_tables: HashMap::new(),
            bleach: bleach,
            phantom: PhantomData,
        }
    }

    fn ranks(&mut self, samples: Vec<T>) -> Vec<u64>
    where
        T: PartialOrd + Copy,
    {
        let mut vetor = Vec::with_capacity(self.addr_length as usize);
        let mut addresses = Vec::new();
        for i in (0..samples.len()).step_by(self.addr_length as usize) {
            if i + self.addr_length as usize <= samples.len() {
                vetor.append(&mut samples[i..i + self.addr_length as usize].to_vec());
            } else {
                vetor.append(&mut samples[i..samples.len()].to_vec());
            }
            let mut tuples: Vec<(u64, &T)> = vetor
                .iter()
                .enumerate()
                .map(|x| (x.0 as u64, x.1))
                .collect();
            tuples.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap()); // TODO: treat the Option
            let address: Vec<u64> = tuples.iter().map(|a| a.0).collect();
            if !self.rank_tables.contains_key(&address) {
                self.rank_tables.insert(address.clone(), self.last_rank);
                self.last_rank += 1;
            }
            addresses.push(*self.rank_tables.get(&address).unwrap()); // TODO: treat the Option
            vetor.clear();
        }
        addresses
    }

    pub fn train(&mut self, data: Vec<T>, label: String)
    where
        T: PartialOrd + Copy,
    {
        if !self.discs.contains_key(&label) {
            self.discs
                .insert(label.clone(), Discriminator::new(self.number_of_hashtables));
        }

        let samples = self.mapping.clone();
        let samples = samples
            .iter()
            .map(|&i| *data.get(i as usize).unwrap())
            .collect();
        let addresses: Vec<u64> = self.ranks(samples);
        let disc = self.discs.get_mut(&label).unwrap();
        disc.train(addresses);
    }

    pub fn classify(&mut self, data: Vec<T>) -> (String, f64, f64)
    where
        T: PartialOrd + Copy,
    {
        let samples = self.mapping.clone();
        let samples = samples
            .iter()
            .map(|&i| *data.get(i as usize).unwrap())
            .collect();
        let addresses: Vec<u64> = self.ranks(samples);
        let discs = &self.discs;
        let mut votes: Vec<(String, (u64, u64))> = discs
            .iter()
            .map(|d| (d.0.to_string(), d.1.classify(&addresses, self.bleach)))
            .collect();
        votes.sort_by(|a, b| (a.1).0.partial_cmp(&(b.1).0).unwrap());

        let biggest = votes.len().checked_sub(1).map(|i| &votes[i]).unwrap();
        let second_biggest = votes.len().checked_sub(2).map(|i| &votes[i]).unwrap();

        (
            biggest.0.clone(),                                        // elected label
            (biggest.1 .0 as f64 / self.number_of_hashtables as f64), // "acc"
            (biggest.1 .0 as f64 - second_biggest.1 .0 as f64) / biggest.1 .0 as f64, // confidence
        )
    }
}

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn test_lib_ranks() {
        // this test verifies that ranks is able to push address to rank_tables
        let mut wis = Wisard::<u8>::new(28, 8, 0);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        wis.ranks(samples);
        assert!(!wis.rank_tables.is_empty());
    }
    #[test]
    fn test_lib_rank_table_length() {
        // this test ensures that the same addresses aren't pushed into the rank_tables
        // repeatedly
        let mut wis = Wisard::<u64>::new(28, 8, 0);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        wis.ranks(samples);
        let length1 = wis.rank_tables.len();
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        wis.ranks(samples);
        let length2 = wis.rank_tables.len();
        assert_eq!(length1, length2);
    }

    #[test]
    fn test_lib_rank_addresses() {
        // this test verifies that for each new piece of data, a correct rank is attributed
        let mut wis = Wisard::<u16>::new(28, 8, 0);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        let addresses = wis.ranks(samples);
        assert_eq!(addresses, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }
    #[test]
    fn test_lib_rank_different_addresses() {
        // this test verifies that small changes in data get close addresses
        let mut wis = Wisard::<u8>::new(28, 8, 0);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        let addresses = wis.ranks(samples);
        assert_eq!(addresses, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 205,
        ];
        let addresses = wis.ranks(samples);
        assert_eq!(addresses, vec![0, 1, 2, 3, 4, 5, 6, 7, 9]);
    }
}
