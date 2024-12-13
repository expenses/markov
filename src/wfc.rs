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
    fn clear(&mut self) {
        self.queue.clear();
        self.sets.clear();
    }

    fn insert_set(&mut self, p: P, set: IndexSet<T>) {
        self.queue.push(p);
        self.sets.insert(p, set);
    }

    // I'd prefer to return an Option<Set> here but that needs the
    // polonius borrow checker to be enabled (-Zpolonius)
    fn peek<O, F: FnOnce(&IndexSet<T>) -> O>(&mut self, func: F) -> Option<O> {
        while let Some(p) = self.queue.peek_mut() {
            if let hash_map::Entry::Occupied(set) = self.sets.entry(*p) {
                if !set.get().is_empty() {
                    return Some(func(set.into_mut()));
                } else {
                    set.remove();
                }
            }

            binary_heap::PeekMut::pop(p);
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

pub type Wave = u64;

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

fn tile_list_from_wave<Wave: WaveNum, const BITS: usize>(
    value: Wave,
) -> arrayvec::ArrayVec<u8, { BITS }> {
    let mut tile_list = arrayvec::ArrayVec::new();

    for i in 0..BITS {
        if ((value >> i) & Wave::one()) == Wave::zero() {
            continue;
        }

        tile_list.push(i as _);
    }

    tile_list
}

pub trait WaveNum:
    std::ops::BitOrAssign + std::ops::BitAndAssign + Default + num_traits::int::PrimInt
{
}

impl<T: std::ops::BitOrAssign + std::ops::BitAndAssign + Default + num_traits::int::PrimInt> WaveNum
    for T
{
}

#[repr(transparent)]
#[derive(Default, Debug, Clone)]
struct Tile<Wave> {
    connections: [Wave; 6],
}

impl<Wave: WaveNum> Tile<Wave> {
    fn connect(&mut self, other: usize, axis: Axis) {
        self.connections[axis as usize] |= Wave::one().shl(other);
    }
}

#[derive(Default, Clone)]
pub struct Tileset<Wave: WaveNum, const BITS: usize> {
    tiles: arrayvec::ArrayVec<Tile<Wave>, { BITS }>,
    probabilities: arrayvec::ArrayVec<f32, { BITS }>,
}

impl<Wave: WaveNum, const BITS: usize> Tileset<Wave, BITS> {
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

    pub fn into_wfc(mut self, size: (u32, u32, u32)) -> Wfc<Wave, BITS> {
        let mut sum = 0.0;
        for &prob in &self.probabilities {
            sum += prob;
        }
        for prob in &mut self.probabilities {
            *prob /= sum;
        }

        let (width, height, depth) = size;
        let mut wfc = Wfc {
            tiles: self.tiles,
            probabilities: self.probabilities,
            array: vec![Wave::zero(); (width * height * depth) as usize],
            width,
            height,
            stack: Vec::new(),
            entropy_to_indices: Default::default(),
        };

        wfc.reset();

        wfc
    }

    pub fn create_wfc(&self, size: (u32, u32, u32)) -> Wfc<Wave, BITS> {
        self.clone().into_wfc(size)
    }

    pub fn num_tiles(&self) -> usize {
        self.tiles.len()
    }
}

pub struct Wfc<Wave: WaveNum, const BITS: usize> {
    tiles: arrayvec::ArrayVec<Tile<Wave>, { BITS }>,
    probabilities: arrayvec::ArrayVec<f32, { BITS }>,
    array: Vec<Wave>,
    width: u32,
    height: u32,
    stack: Vec<(u32, Wave)>,
    entropy_to_indices: SetQueue<u32, Reverse<OrderedFloat<f32>>>,
}

impl<Wave: WaveNum, const BITS: usize> Wfc<Wave, BITS> {
    pub fn reset(&mut self) {
        self.stack.clear();
        let wave = Wave::max_value() >> (BITS - self.tiles.len());
        for value in self.array.iter_mut() {
            *value = wave;
        }
        self.entropy_to_indices.clear();
        let mut set = IndexSet::new();
        for i in 0..self.array.len() {
            set.insert(i as u32);
        }
        self.entropy_to_indices.insert_set(
            Reverse(OrderedFloat(self.calculate_shannon_entropy(wave))),
            set,
        );
    }

    pub fn num_tiles(&self) -> usize {
        self.tiles.len()
    }

    pub fn calculate_shannon_entropy(&self, wave: Wave) -> f32 {
        let mut sum = 0.0;
        for i in tile_list_from_wave::<_, BITS>(wave) {
            let prob = self.probabilities[i as usize];

            if prob <= 0.0 {
                continue;
            }

            sum -= prob * prob.log2();
        }
        sum
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn depth(&self) -> u32 {
        self.array.len() as u32 / self.width() / self.height()
    }

    pub fn find_lowest_entropy(&mut self, rng: &mut SmallRng) -> Option<(u32, u8)> {
        self.entropy_to_indices.peek(|set| {
            let index = rng.gen_range(0..set.len());
            let index = *set.get_index(index).unwrap();

            let value = self.array[index as usize];

            let mut rolling_probability: arrayvec::ArrayVec<_, { BITS }> = Default::default();

            let list = tile_list_from_wave::<_, BITS>(value);

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

            (index, tile)
        })
    }

    pub fn collapse_all_reset_on_contradiction(&mut self, rng: &mut SmallRng) -> u32 {
        let mut attempts = 1;
        while let Some((index, tile)) = self.find_lowest_entropy(rng) {
            if self.collapse(index, tile) {
                self.reset();
                attempts += 1
            }
        }

        attempts
    }

    pub fn collapse_all(&mut self, rng: &mut SmallRng) -> bool {
        let mut any_contradictions = false;
        while let Some((index, tile)) = self.find_lowest_entropy(rng) {
            if self.collapse(index, tile) {
                any_contradictions = true;
            }
        }

        any_contradictions
    }

    pub fn collapse(&mut self, index: u32, tile: u8) -> bool {
        self.partial_collapse(index, Wave::one().shl(tile as _))
    }

    pub fn partial_collapse(&mut self, index: u32, remaining_possible_states: Wave) -> bool {
        self.stack.clear();
        self.stack.push((index, remaining_possible_states));
        let mut any_contradictions = false;

        while let Some((index, remaining_possible_states)) = self.stack.pop() {
            let old = self.array[index as usize];
            self.array[index as usize] &= remaining_possible_states;
            let new = self.array[index as usize];

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

            if new == Wave::zero() {
                any_contradictions = true;
                continue;
            }

            if new.count_ones() > 1 {
                let _val = self.entropy_to_indices.insert(
                    Reverse(OrderedFloat(self.calculate_shannon_entropy(new))),
                    index,
                );
                debug_assert!(_val);
            }

            let new_tiles = tile_list_from_wave::<_, BITS>(new);

            for axis in Axis::ALL {
                let (mut x, mut y, mut z) =
                    decompose(index as _, self.width() as _, self.height() as _);
                match axis {
                    Axis::X if x < self.width() as usize - 1 => x += 1,
                    Axis::Y if y < self.height() as usize - 1 => y += 1,
                    Axis::Z if z < self.depth() as usize - 1 => z += 1,
                    Axis::NegX if x > 0 => x -= 1,
                    Axis::NegY if y > 0 => y -= 1,
                    Axis::NegZ if z > 0 => z -= 1,
                    _ => continue,
                };

                let index = compose(x, y, z, self.width() as _, self.height() as _) as u32;

                let mut valid = Wave::zero();

                for &tile in new_tiles.iter() {
                    valid |= self.tiles[tile as usize].connections[axis as usize];
                }

                self.stack.push((index, valid));
            }
        }

        any_contradictions
    }

    pub fn values(&self) -> Vec<u8> {
        let mut values = vec![0; self.array.len()];
        self.set_values(&mut values);
        values
    }

    pub fn set_values(&self, values: &mut [u8]) {
        self.array.iter().zip(values).for_each(|(wave, value)| {
            *value = if wave.count_ones() == 1 {
                wave.trailing_zeros() as u8
            } else {
                u8::max_value()
            }
        });
    }

    #[cfg(test)]
    fn all_collapsed(&self) -> bool {
        self.array.iter().all(|&value| value.count_ones() == 1)
    }
}

#[cfg(test)]
use rand::SeedableRng;

#[test]
fn normal() {
    let mut rng = SmallRng::from_entropy();

    let mut tileset = Tileset::<u8, 8>::default();
    let sea = tileset.add(1.0);
    let beach = tileset.add(0.5);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis::ALL);
    tileset.connect(sea, beach, &Axis::ALL);
    tileset.connect(beach, beach, &Axis::ALL);
    tileset.connect(beach, grass, &Axis::ALL);
    tileset.connect(grass, grass, &Axis::ALL);

    assert_eq!(tileset.tiles[sea].connections, [3; 6]);

    let mut wfc = tileset.into_wfc((100, 1000, 1));

    assert!(!wfc.all_collapsed());
    assert!(!wfc.collapse_all(&mut rng));
    assert!(
        wfc.all_collapsed(),
        "failed to collapse: {:?}",
        &wfc.array.iter().map(|v| v.count_ones()).collect::<Vec<_>>()
    );
}

#[test]
fn verticals() {
    let mut rng = SmallRng::from_entropy();

    let mut tileset = Tileset::<u64, 64>::default();
    let air = tileset.add(1.0);
    let solid = tileset.add(1.0);
    tileset.connect(air, air, &Axis::ALL);
    tileset.connect(solid, solid, &Axis::ALL);
    // solid cant be above air
    tileset.connect(
        solid,
        air,
        &[Axis::X, Axis::Y, Axis::Z, Axis::NegX, Axis::NegY],
    );

    let mut wfc = tileset.into_wfc((50, 50, 50));

    assert!(!wfc.all_collapsed());
    assert!(!wfc.collapse_all(&mut rng));
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

    let mut tileset = Tileset::<u64, 64>::default();
    let empty = tileset.add(0.0);
    let ground = tileset.add(1.0);
    tileset.connect(ground, ground, &[Axis::X, Axis::Y]);
    let stairs_top = tileset.add(1.0);
    let stairs_bottom = tileset.add(10.0);
    tileset.connect(stairs_top, stairs_bottom, &[Axis::X, Axis::NegZ]);
    tileset.connect(stairs_top, ground, &[Axis::X]);
    tileset.connect(stairs_bottom, ground, &[Axis::NegX]);
    //tileset.connect(solid, solid, &Axis::ALL);

    tileset.connect_to_all(empty);

    let mut wfc = tileset.into_wfc((5, 5, 5));

    assert!(!wfc.collapse_all(&mut rng));
    assert!(wfc.all_collapsed(),);
}

#[test]
fn broken() {
    let mut rng = SmallRng::from_entropy();

    let mut tileset = Tileset::<u64, 64>::default();

    let sea = tileset.add(1.0);
    let beach = tileset.add(1.0);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis::ALL);
    tileset.connect(sea, beach, &Axis::ALL);
    //tileset.connect(beach, beach, &Axis::ALL);
    tileset.connect(beach, grass, &Axis::ALL);
    tileset.connect(grass, grass, &Axis::ALL);

    assert_eq!(tileset.tiles[sea].connections, [3; 6]);

    // Wait until there's a collapse failure due to beaches not being able to connect to beaches.
    loop {
        let mut wfc = tileset.create_wfc((10, 10, 1));

        assert!(!wfc.all_collapsed());

        if wfc.collapse_all(&mut rng) {
            assert!(!wfc.all_collapsed());
            // Make sure that at least one state has collapsed properly (aka that the error hasn't spread).
            assert!(wfc.array.iter().any(|&v| v.count_ones() == 1));
            break;
        }
    }
}

#[test]
fn pipes() {
    let mut rng = SmallRng::from_entropy();

    let mut tileset = Tileset::<u16, 16>::default();

    let empty = tileset.add(1.0);
    let pipe_x = tileset.add(1.0);
    let pipe_y = tileset.add(1.0);
    let t = tileset.add(1.0);
    tileset.connect(empty, empty, &Axis::ALL);
    tileset.connect(pipe_x, pipe_x, &Axis::ALL);
    tileset.connect(pipe_y, pipe_y, &Axis::ALL);
    tileset.connect(empty, pipe_x, &[Axis::X, Axis::NegX]);
    tileset.connect(empty, pipe_y, &[Axis::Y, Axis::NegY]);
    tileset.connect(empty, t, &[Axis::Z, Axis::NegZ, Axis::NegY]);
    tileset.connect(t, pipe_y, &[Axis::Y]);
    tileset.connect(t, pipe_y, &[Axis::X, Axis::NegX]);

    tileset
        .into_wfc((10, 10, 10))
        .collapse_all_reset_on_contradiction(&mut rng);
}
