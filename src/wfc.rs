use crate::arrays::{compose, decompose};
use indexmap::IndexSet;
use ordered_float::OrderedFloat;
use rand::{rngs::SmallRng, Rng};
use std::cmp::Ord;
use std::cmp::Reverse;
use std::collections::{binary_heap, hash_map, BinaryHeap, HashMap};
use std::hash::Hash;

#[derive(Default)]
struct SetQueue<T, P: Ord> {
    queue: BinaryHeap<P>,
    sets: HashMap<P, IndexSet<T>>,
}

impl<T: Hash + Eq, P: Copy + Ord + Hash> SetQueue<T, P> {
    fn insert_set(&mut self, p: P, set: IndexSet<T>) {
        self.queue.push(p);
        self.sets.insert(p, set);
    }

    // Peek only once. Needs the new polonius borrow checker to be enabled
    // for a looping version (-Zpolonius)
    fn try_peek(&mut self) -> Option<Option<&IndexSet<T>>> {
        if let Some(p) = self.queue.peek_mut() {
            if let hash_map::Entry::Occupied(set) = self.sets.entry(*p) {
                if !set.get().is_empty() {
                    return Some(Some(set.into_mut()));
                } else {
                    set.remove();
                }
            }

            binary_heap::PeekMut::pop(p);
            return Some(None);
        }

        None
    }

    fn insert(&mut self, p: P, value: T) -> bool {
        let set = match self.sets.entry(p) {
            hash_map::Entry::Occupied(set) => set.into_mut(),
            hash_map::Entry::Vacant(set) => {
                self.queue.push(p);
                set.insert(Default::default())
            }
        };
        set.insert(value)
    }

    fn remove(&mut self, p: P, value: &T) -> bool {
        if let Some(set) = self.sets.get_mut(&p) {
            set.swap_remove(value)
        } else {
            false
        }
    }
}

type Wave = u64;

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
    NegX = 3,
    NegY = 4,
    NegZ = 5,
}

impl Axis {
    pub const ALL: [Self; 6] = [
        Self::X,
        Self::Y,
        Self::Z,
        Self::NegX,
        Self::NegY,
        Self::NegZ,
    ];

    pub fn opp(&self) -> Axis {
        match self {
            Self::X => Self::NegX,
            Self::Y => Self::NegY,
            Self::Z => Self::NegZ,
            Self::NegX => Self::X,
            Self::NegY => Self::Y,
            Self::NegZ => Self::Z,
        }
    }
}

fn tile_list_from_wave(value: Wave) -> arrayvec::ArrayVec<u8, { Wave::BITS as _ }> {
    let mut tile_list = arrayvec::ArrayVec::new();

    for i in 0..Wave::BITS {
        if ((value >> i) & 1) == 0 {
            continue;
        }

        tile_list.push(i as _);
    }

    tile_list
}

#[repr(transparent)]
#[derive(Default, Debug)]
struct Tile {
    connections: [Wave; 6],
}

impl Tile {
    fn connect(&mut self, other: usize, axis: Axis) {
        self.connections[axis as usize] |= 1 << other;
    }
}

/*#[derive(Default)]
struct Tileset {
    tiles: arrayvec::ArrayVec<Tile, { Wave::BITS as _ }>,
    probabilities: arrayvec::ArrayVec<f32, { Wave::BITS as _ }>,
}

impl Tileset {
    fn add(&mut self, probability: f32) -> usize {
        let index = self.tiles.len();
        self.tiles.push(Tile::default());
        self.probabilities.push(probability);
        index
    }

    fn connect(&mut self, from: usize, to: usize, axises: &[Axis]) {
        for &axis in axises {
            self.tiles[from].connect(to, axis);
            self.tiles[to].connect(from, axis.opp());
        }
    }

    fn connect_to_all(&mut self, tile: usize) {
        for other in 0..self.tiles.len() {
            self.connect(tile, other, &Axis::ALL)
        }
    }
}*/

pub struct Wfc {
    tiles: arrayvec::ArrayVec<Tile, { Wave::BITS as _ }>,
    probabilities: arrayvec::ArrayVec<f32, { Wave::BITS as _ }>,
    array: Vec<Wave>,
    width: usize,
    height: usize,
    stack: Vec<(usize, Wave)>,
    entropy_to_indices: SetQueue<usize, Reverse<OrderedFloat<f32>>>,
}

impl Wfc {
    pub fn new(size: (usize, usize, usize)) -> Self {
        let (width, height, depth) = size;
        Self {
            tiles: Default::default(),
            probabilities: Default::default(),
            array: vec![0; width * height * depth],
            width,
            height,
            stack: Vec::new(),
            entropy_to_indices: Default::default(),
        }
    }

    pub fn calculate_shannon_entropy(&self, wave: Wave) -> f32 {
        let mut sum = 0.0;
        for i in tile_list_from_wave(wave) {
            let prob = self.probabilities[i as usize];

            if prob <= 0.0 {
                continue;
            }

            sum -= prob * prob.log2();
        }
        sum
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }

    pub fn depth(&self) -> usize {
        self.array.len() / self.width / self.height
    }

    pub fn add(&mut self, probability: f32) -> usize {
        let index = self.tiles.len();
        self.tiles.push(Tile::default());
        self.probabilities.push(probability);
        index
    }

    pub fn connect(&mut self, from: usize, to: usize, axises: &[Axis]) {
        for &axis in axises {
            self.tiles[from].connect(to, axis);
            self.tiles[to].connect(from, axis.opp());
        }
    }

    pub fn connect_to_all(&mut self, tile: usize) {
        for other in 0..self.tiles.len() {
            self.connect(tile, other, &Axis::ALL)
        }
    }

    pub fn setup_state(&mut self) {
        let mut sum = 0.0;
        for &prob in &self.probabilities {
            sum += prob;
        }
        for prob in &mut self.probabilities {
            *prob /= sum;
        }

        let wave = Wave::MAX >> (Wave::BITS as usize - self.tiles.len());
        for value in &mut self.array {
            *value = wave;
        }

        let mut set = IndexSet::new();

        for i in 0..self.array.len() {
            set.insert(i);
        }

        self.entropy_to_indices.insert_set(
            Reverse(OrderedFloat(self.calculate_shannon_entropy(wave))),
            set,
        );
    }

    pub fn find_lowest_entropy(&mut self, rng: &mut SmallRng) -> Option<(usize, u8)> {
        let lowest_entropy_set = loop {
            if let Some(v) = self.entropy_to_indices.try_peek() {
                match v {
                    None => {}
                    Some(set) => break set,
                }
            } else {
                return None;
            }
        };

        let index = rng.gen_range(0..lowest_entropy_set.len());
        let index = *lowest_entropy_set.get_index(index).unwrap();

        let value = self.array[index];

        let mut rolling_probability: arrayvec::ArrayVec<_, { Wave::BITS as _ }> =
            Default::default();

        let list = tile_list_from_wave(value);

        let mut sum = 0.0;
        for &tile in &list {
            sum += self.probabilities[tile as usize];
            rolling_probability.push(OrderedFloat(sum));
        }
        let num = rng.gen_range(0.0..=rolling_probability.last().unwrap().0);
        let list_index = match rolling_probability.binary_search(&OrderedFloat(num)) {
            Ok(index) => index,
            Err(index) => index,
        };

        let tile = list[list_index];

        Some((index, tile))
    }

    pub fn collapse_all(&mut self, rng: &mut SmallRng) {
        while let Some((index, tile)) = self.find_lowest_entropy(rng) {
            self.collapse(index, tile);
        }
    }

    pub fn collapse(&mut self, index: usize, tile: u8) {
        self.set(index, 1 << tile);
    }

    fn set(&mut self, index: usize, remaining_possible_states: Wave) {
        self.stack.clear();
        self.stack.push((index, remaining_possible_states));

        while let Some((index, remaining_possible_states)) = self.stack.pop() {
            let old = self.array[index];
            self.array[index] &= remaining_possible_states;
            let new = self.array[index];

            if old == new {
                continue;
            }

            if old.count_ones() > 1 {
                let _val = self.entropy_to_indices.remove(
                    Reverse(OrderedFloat(self.calculate_shannon_entropy(old))),
                    &index,
                );
                debug_assert!(_val);
            }

            if new == 0 {
                continue;
            }

            if new.count_ones() > 1 {
                let _val = self.entropy_to_indices.insert(
                    Reverse(OrderedFloat(self.calculate_shannon_entropy(new))),
                    index,
                );
                debug_assert!(_val);
            }

            let new_tiles = tile_list_from_wave(new);

            for axis in Axis::ALL {
                let (mut x, mut y, mut z) = decompose(index, self.width, self.height);
                match axis {
                    Axis::X if x < self.width - 1 => x += 1,
                    Axis::Y if y < self.height - 1 => y += 1,
                    Axis::Z if z < self.depth() - 1 => z += 1,
                    Axis::NegX if x > 0 => x -= 1,
                    Axis::NegY if y > 0 => y -= 1,
                    Axis::NegZ if z > 0 => z -= 1,
                    _ => continue,
                };

                let index = compose(x, y, z, self.width, self.height);

                let mut valid = 0;

                for &tile in new_tiles.iter() {
                    valid |= self.tiles[tile as usize].connections[axis as usize];
                }

                self.stack.push((index, valid));
            }
        }
    }

    pub fn all_collapsed(&self) -> bool {
        self.array.iter().all(|&value| value.count_ones() == 1)
    }

    pub fn values(&self) -> Vec<u8> {
        self.array
            .iter()
            .map(|&value| value.trailing_zeros() as u8)
            .collect()
    }
}

#[cfg(test)]
use rand::SeedableRng;

#[test]
fn normal() {
    let mut rng = SmallRng::from_entropy();

    let mut wfc = Wfc::new((100, 1000, 1));
    let sea = wfc.add(1.0);
    let beach = wfc.add(0.5);
    let grass = wfc.add(1.0);
    wfc.connect(sea, sea, &Axis::ALL);
    wfc.connect(sea, beach, &Axis::ALL);
    wfc.connect(beach, beach, &Axis::ALL);
    wfc.connect(beach, grass, &Axis::ALL);
    wfc.connect(grass, grass, &Axis::ALL);

    assert_eq!(wfc.tiles[sea].connections, [3; 6]);

    wfc.setup_state();

    assert!(!wfc.all_collapsed());
    wfc.collapse_all(&mut rng);
    assert!(
        wfc.all_collapsed(),
        "failed to collapse: {:?}",
        &wfc.array.iter().map(|v| v.count_ones()).collect::<Vec<_>>()
    );
}

#[test]
fn verticals() {
    let mut rng = SmallRng::from_entropy();

    let mut wfc = Wfc::new((50, 50, 50));
    let air = wfc.add(1.0);
    let solid = wfc.add(1.0);
    wfc.connect(air, air, &Axis::ALL);
    wfc.connect(solid, solid, &Axis::ALL);
    // solid cant be above air
    wfc.connect(
        solid,
        air,
        &[Axis::X, Axis::Y, Axis::Z, Axis::NegX, Axis::NegY],
    );

    wfc.setup_state();

    assert!(!wfc.all_collapsed());
    wfc.collapse_all(&mut rng);
    assert!(
        wfc.all_collapsed(),
        "{:?}",
        &wfc.array.iter().map(|v| v.count_ones()).collect::<Vec<_>>()
    );
    let _v = wfc.values();
    //panic!("{:?}",v);
}

#[test]
fn stairs() {
    let mut rng = SmallRng::from_entropy();

    let mut wfc = Wfc::new((5, 5, 5));
    let empty = wfc.add(0.0);
    let ground = wfc.add(1.0);
    wfc.connect(ground, ground, &[Axis::X, Axis::Y]);
    let stairs_top = wfc.add(1.0);
    let stairs_bottom = wfc.add(10.0);
    wfc.connect(stairs_top, stairs_bottom, &[Axis::X, Axis::NegZ]);
    wfc.connect(stairs_top, ground, &[Axis::X]);
    wfc.connect(stairs_bottom, ground, &[Axis::NegX]);
    //wfc.connect(solid, solid, &Axis::ALL);

    wfc.connect_to_all(empty);
    wfc.setup_state();

    wfc.collapse_all(&mut rng);
    assert!(wfc.all_collapsed(),);
}

#[test]
fn broken() {
    let mut rng = SmallRng::from_entropy();

    // Wait until there's a collapse failure due to beaches not being able to connect to beaches.
    loop {
        let mut wfc = Wfc::new((10, 10, 1));
        let sea = wfc.add(1.0);
        let beach = wfc.add(1.0);
        let grass = wfc.add(1.0);
        wfc.connect(sea, sea, &Axis::ALL);
        wfc.connect(sea, beach, &Axis::ALL);
        //wfc.connect(beach, beach, &Axis::ALL);
        wfc.connect(beach, grass, &Axis::ALL);
        wfc.connect(grass, grass, &Axis::ALL);

        assert_eq!(wfc.tiles[sea].connections, [3; 6]);

        wfc.setup_state();

        assert!(!wfc.all_collapsed());
        wfc.collapse_all(&mut rng);

        if !wfc.all_collapsed() {
            // Make sure that at least one state has collapsed properly (aka that the error hasn't spread).
            assert!(wfc.array.iter().any(|&v| v.count_ones() == 1));
            break;
        }
    }
}
