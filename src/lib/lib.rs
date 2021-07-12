use image::{GrayImage, RgbImage};
use std::collections::HashMap;

pub struct Discriminator {
    number_of_hashtables: u16,
    h_rams: Vec<HashMap<u64, u16>>,
    times_trained: u64,
}

impl Discriminator {
    pub fn new(num: u16) -> Discriminator {
        Discriminator {
            number_of_hashtables: num,
            h_rams: Vec::new(),
            times_trained: 0,
        }
    }

    pub fn train(&mut self, x: Vec<u64>) -> () {
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

pub struct Wisard {
    discs: HashMap<String, Discriminator>,
    addr_length: u64,
    number_of_hashtables: u16,
    mapping: Vec<u64>,
    last_rank: u64,
    rank_tables: HashMap<Vec<u64>, u64>,
}

impl Wisard {
    pub fn new(number_of_hashtables: u16, addr_length: u64) -> Wisard {
        Wisard {
            discs: HashMap::new(),
            addr_length: addr_length,
            number_of_hashtables: number_of_hashtables,
            mapping: (0..addr_length as u64 * number_of_hashtables as u64).collect::<Vec<u64>>(),
            last_rank: 0,
            rank_tables: HashMap::new(),
        }
    }

    fn ranks<T: PartialOrd + Clone>(&mut self, samples: Vec<T>) -> Vec<u64> {
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

    pub fn train(mut self, img: GrayImage, label: String) {
        if !self.discs.contains_key(&label) {
            self.discs
                .insert(label.clone(), Discriminator::new(self.number_of_hashtables));
        }

        let flat_image = img.into_flat_samples().to_vec().samples;
        let samples: Vec<u8> = flat_image
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.mapping.contains(&(i as u64)))
            .map(|(_, e)| *e)
            .collect();
        let addresses: Vec<u64> = self.ranks(samples);
        let disc = self.discs.get_mut(&label).unwrap();
        disc.train(addresses);
    }

    pub fn classiy(self, img: GrayImage) {
        unimplemented!();
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_ranks() {
        let mut wis = Wisard::new(28, 8);
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
    fn test_rank_table_length() {
        let mut wis = Wisard::new(28, 8);
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
    fn test_rank_addresses() {
        let mut wis = Wisard::new(28, 8);
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
    fn test_rank_differente_addresses() {
        let mut wis = Wisard::new(28, 8);
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
