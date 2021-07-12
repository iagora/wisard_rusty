use image::{Rgb, RgbImage};

pub struct Wisard {
    discs: Vec<(String, String)>,
    addr_length: u64,
    number_of_hashtables: u16,
    mapping: Vec<u64>,
    last_rank: u64,
    rank_tables: Vec<Vec<usize>>,
}

impl Wisard {
    pub fn new(number_of_hashtables: u16, addr_length: u64) -> Self {
        Self {
            discs: Vec::new(),
            addr_length: addr_length,
            number_of_hashtables: number_of_hashtables,
            mapping: (0..addr_length as u64 * number_of_hashtables as u64).collect::<Vec<u64>>(),
            last_rank: 0,
            rank_tables: Vec::new(),
        }
    }

    fn ranks<T: PartialOrd + Clone>(&mut self, samples: Vec<T>) {
        // find a way to change to generic
        let mut vetor = Vec::with_capacity(self.addr_length as usize);
        for i in (0..samples.len()).step_by(self.addr_length as usize) {
            if i + self.addr_length as usize <= samples.len() {
                vetor.append(&mut samples[i..i + self.addr_length as usize].to_vec());
            } else {
                vetor.append(&mut samples[i..samples.len()].to_vec());
            }
            let mut tuples: Vec<(usize, &T)> = vetor.iter().enumerate().collect();
            tuples.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
            let adress: Vec<usize> = tuples.iter().map(|a| a.0).collect();
            self.rank_tables.push(adress);
            vetor.clear();
        }
    }

    pub fn train(self, img: RgbImage, label: String) {
        unimplemented!();
    }

    pub fn classiy(self, img: RgbImage) {
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
}
