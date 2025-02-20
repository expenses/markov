use fnv::FnvBuildHasher;
use ordered_float::OrderedFloat;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::cmp::Ord;
use std::cmp::Reverse;
use std::collections::{binary_heap, hash_map, BinaryHeap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

#[derive(Clone)]
struct IndexSet<T> {
    values: Vec<T>,
    indices: hashbrown::HashTable<u32>,
}

impl<T> Default for IndexSet<T> {
    fn default() -> Self {
        Self {
            values: Default::default(),
            indices: Default::default(),
        }
    }
}

impl<T: Hash + Eq + Clone + Debug> IndexSet<T> {
    fn new_from_vec(values: Vec<T>) -> Self {
        let mut indices = hashbrown::HashTable::with_capacity(values.len());

        for (index, value) in values.iter().cloned().enumerate() {
            use std::hash::BuildHasher;
            let hasher = FnvBuildHasher::default();
            // Only used when resizing
            let hasher_fn = |_: &u32| unreachable!();
            indices.insert_unique(hasher.hash_one(&value), index as u32, hasher_fn);
        }

        Self { values, indices }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn insert(&mut self, value: T) -> bool {
        use std::hash::BuildHasher;
        let hasher = FnvBuildHasher::default();

        let index = self.values.len() as u32;

        let hasher_fn =
            |&val: &u32| unsafe { hasher.hash_one(self.values.get_unchecked(val as usize)) };

        match self
            .indices
            .entry(hasher.hash_one(&value), |other| *other == index, hasher_fn)
        {
            hashbrown::hash_table::Entry::Vacant(slot) => {
                slot.insert(index);
                self.values.push(value);
                true
            }
            _ => false,
        }
    }

    fn get_index(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    fn swap_remove(&mut self, value: &T) -> bool {
        use std::hash::BuildHasher;
        let hasher = FnvBuildHasher::default();
        // Search in the table for the value
        let index = if let Ok(entry) = self
            .indices
            .find_entry(hasher.hash_one(value), |&other| unsafe {
                self.values.get_unchecked(other as usize) == value
            }) {
            let (index, _) = entry.remove();
            index
        } else {
            return false;
        };
        // Swap remove it.
        self.values.swap_remove(index as _);
        // If the vec still has items, we need to change it's value.
        if let Some(value) = self.values.get(index as usize) {
            use std::hash::BuildHasher;
            let hasher = FnvBuildHasher::default();
            if let Ok(entry) = self.indices.find_entry(hasher.hash_one(value), |&other| {
                other == self.values.len() as u32
            }) {
                *entry.into_mut() = index;
            }
        }
        true
    }
}

#[derive(Default, Clone)]
struct SetQueue<T, P: Ord> {
    queue: BinaryHeap<P>,
    sets: HashMap<P, IndexSet<T>, FnvBuildHasher>,
}

impl<T: Hash + Eq + Clone + Debug, P: Copy + Ord + Hash> SetQueue<T, P> {
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
            hash_map::Entry::Vacant(slot) => {
                self.queue.push(p);
                slot.insert(Default::default())
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
    std::ops::BitOrAssign
    + std::ops::BitAndAssign
    + Default
    + num_traits::int::PrimInt
    + Debug
    + Send
    + Sync
{
}

impl<
        T: std::ops::BitOrAssign
            + std::ops::BitAndAssign
            + Default
            + num_traits::int::PrimInt
            + Debug
            + Send
            + Sync,
    > WaveNum for T
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
    #[inline]
    pub fn add(&mut self, probability: f32) -> usize {
        let index = self.tiles.len();
        self.tiles.push(Tile::default());
        self.probabilities.push(probability);
        index
    }

    #[inline]
    pub fn connect(&mut self, from: usize, to: usize, axises: &[Axis]) {
        for &axis in axises {
            self.tiles[from].connect(to, axis);
            self.tiles[to].connect(from, axis.opp());
        }
    }

    #[inline]
    pub fn connect_to_all(&mut self, tile: usize) {
        for other in 0..self.tiles.len() {
            self.connect(tile, other, &Axis::ALL)
        }
    }

    fn normalize_probabilities(&mut self) {
        let mut sum = 0.0;
        for &prob in &self.probabilities {
            sum += prob;
        }
        for prob in &mut self.probabilities {
            *prob /= sum;
        }
    }

    #[inline]
    pub fn into_wfc<E: Entropy>(mut self, size: (u32, u32, u32)) -> Wfc<Wave, E, BITS> {
        self.normalize_probabilities();

        let (width, height, depth) = size;
        let mut wfc = Wfc {
            tiles: self.tiles,
            probabilities: self.probabilities,
            state: State {
                array: vec![Wave::zero(); (width * height * depth) as usize],
                entropy_to_indices: Default::default(),
            },
            initial_state: State::default(),
            width,
            height,
            stack: Vec::new(),
        };

        wfc.reset();

        wfc
    }

    #[inline]
    pub fn into_wfc_with_initial_state<E: Entropy>(
        mut self,
        size: (u32, u32, u32),
        array: Vec<Wave>,
    ) -> Wfc<Wave, E, BITS> {
        self.normalize_probabilities();

        let (width, height, depth) = size;
        let mut wfc = Wfc {
            tiles: self.tiles,
            probabilities: self.probabilities,
            state: State {
                array: vec![Wave::zero(); (width * height * depth) as usize],
                entropy_to_indices: Default::default(),
            },
            initial_state: State {
                array,
                entropy_to_indices: Default::default(),
            },
            width,
            height,
            stack: Vec::new(),
        };

        wfc.collapse_initial_state();

        wfc
    }

    #[inline]
    pub fn create_wfc<E: Entropy>(&self, size: (u32, u32, u32)) -> Wfc<Wave, E, BITS> {
        self.clone().into_wfc(size)
    }

    #[inline]
    pub fn create_wfc_with_initial_state<E: Entropy>(
        &self,
        size: (u32, u32, u32),
        array: Vec<Wave>,
    ) -> Wfc<Wave, E, BITS> {
        self.clone().into_wfc_with_initial_state(size, array)
    }

    #[inline]
    pub fn num_tiles(&self) -> usize {
        self.tiles.len()
    }

    #[inline]
    pub fn initial_wave(&self) -> Wave {
        Wave::max_value() >> (BITS - self.tiles.len())
    }
}

pub trait Entropy: Default + Send + Clone {
    type Type: Ord + Clone + Copy + Default + Hash + Send;
    fn calculate<Wave: WaveNum, const BITS: usize>(probabilities: &[f32], wave: Wave)
        -> Self::Type;
}

#[derive(Clone, Default)]
pub struct ShannonEntropy;

impl Entropy for ShannonEntropy {
    type Type = OrderedFloat<f32>;

    #[inline]
    fn calculate<Wave: WaveNum, const BITS: usize>(
        probabilities: &[f32],
        wave: Wave,
    ) -> Self::Type {
        let mut sum = 0.0;
        for i in tile_list_from_wave::<_, BITS>(wave) {
            let prob = probabilities[i as usize];

            if prob <= 0.0 {
                continue;
            }

            sum -= prob * prob.log2();
        }
        OrderedFloat(sum)
    }
}

#[derive(Clone, Default)]
pub struct LinearEntropy;

impl Entropy for LinearEntropy {
    type Type = u8;

    #[inline]
    fn calculate<Wave: WaveNum, const BITS: usize>(
        _probabilities: &[f32],
        wave: Wave,
    ) -> Self::Type {
        wave.count_ones() as _
    }
}

#[derive(Clone, Default)]
struct State<Wave: WaveNum, E: Entropy> {
    array: Vec<Wave>,
    entropy_to_indices: SetQueue<u32, Reverse<<E as Entropy>::Type>>,
}

#[derive(Clone)]
pub struct Wfc<Wave: WaveNum, E: Entropy, const BITS: usize> {
    tiles: arrayvec::ArrayVec<Tile<Wave>, { BITS }>,
    probabilities: arrayvec::ArrayVec<f32, { BITS }>,
    state: State<Wave, E>,
    initial_state: State<Wave, E>,
    width: u32,
    height: u32,
    stack: Vec<(u32, Wave)>,
}

impl<Wave: WaveNum, E: Entropy, const BITS: usize> Wfc<Wave, E, BITS> {
    #[inline]
    pub fn initial_wave(&self) -> Wave {
        Wave::max_value() >> (BITS - self.tiles.len())
    }

    #[inline]
    fn collapse_initial_state(&mut self) {
        self.reset_initial();

        let initial_wave = self.initial_wave();

        let mut any_contradictions = false;

        for (i, wave) in std::mem::take(&mut self.initial_state)
            .array
            .iter()
            .copied()
            .enumerate()
        {
            if initial_wave != wave && wave != Wave::zero() {
                any_contradictions |= self.partial_collapse(i as u32, wave);
            }
        }

        assert!(!any_contradictions);

        self.initial_state.clone_from(&self.state);
    }

    fn reset_initial(&mut self) {
        let wave = self.initial_wave();
        for value in self.state.array.iter_mut() {
            *value = wave;
        }
        self.state.entropy_to_indices.clear();
        let set = IndexSet::new_from_vec((0..self.state.array.len() as u32).collect());
        self.state.entropy_to_indices.insert_set(
            Reverse(E::calculate::<Wave, BITS>(
                &self.probabilities,
                self.initial_wave(),
            )),
            set,
        );
    }

    #[inline]
    pub fn reset(&mut self) {
        self.stack.clear();

        if !self.initial_state.array.is_empty() {
            self.state.clone_from(&self.initial_state);
        } else {
            self.reset_initial();
        }
    }

    #[inline]
    pub fn num_tiles(&self) -> usize {
        self.tiles.len()
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    pub fn depth(&self) -> u32 {
        self.state.array.len() as u32 / self.width() / self.height()
    }

    #[inline]
    pub fn find_lowest_entropy(&mut self, rng: &mut SmallRng) -> Option<(u32, u8)> {
        self.state.entropy_to_indices.peek(|set| {
            let index = rng.gen_range(0..set.len());
            let index = *set.get_index(index).unwrap();

            let value = self.state.array[index as usize];

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

    #[inline]
    pub fn collapse_all_reset_on_contradiction_par(&mut self, mut rng: &mut SmallRng) -> u32 {
        let states: Vec<_> = (0..rayon::current_num_threads())
            .map(|_| (self.clone(), SmallRng::from_rng(&mut rng).unwrap()))
            .collect();

        let other_attempts = AtomicU32::new(0);

        let (wfc, local_attempts) =
            find_any_with_early_stop(states, |(mut wfc, mut rng), stop_flag| {
                let mut attempts = 1;
                while let Some((index, tile)) = wfc.find_lowest_entropy(&mut rng) {
                    if stop_flag.load(Ordering::Relaxed) {
                        other_attempts.fetch_add(attempts, Ordering::Relaxed);
                        return None;
                    }
                    if wfc.collapse(index, tile) {
                        if attempts % 1000 == 0 {
                            println!("{}", attempts);
                        }
                        wfc.reset();
                        attempts += 1
                    }
                }
                Some((wfc, attempts))
            })
            .unwrap();

        let total_attempts = local_attempts + other_attempts.load(Ordering::Relaxed);

        println!(
            "Found after {} attempts, ({} thread local)",
            total_attempts, local_attempts,
        );

        *self = wfc;
        total_attempts
    }

    #[inline]
    pub fn collapse_all_reset_on_contradiction(&mut self, rng: &mut SmallRng) -> u32 {
        let mut attempts = 1;
        while let Some((index, tile)) = self.find_lowest_entropy(rng) {
            if self.collapse(index, tile) {
                if attempts % 1000 == 0 {
                    println!("{}", attempts);
                }
                self.reset();
                attempts += 1
            }
        }

        attempts
    }

    #[inline]
    pub fn collapse_all(&mut self, rng: &mut SmallRng) -> bool {
        let mut any_contradictions = false;
        while let Some((index, tile)) = self.find_lowest_entropy(rng) {
            if self.collapse(index, tile) {
                any_contradictions = true;
            }
        }

        any_contradictions
    }

    #[inline]
    pub fn collapse(&mut self, index: u32, tile: u8) -> bool {
        self.partial_collapse(index, Wave::one().shl(tile as _))
    }

    #[inline]
    pub fn partial_collapse(&mut self, index: u32, remaining_possible_states: Wave) -> bool {
        self.stack.clear();
        self.stack.push((index, remaining_possible_states));

        let mut any_contradictions = false;

        while let Some((index, remaining_possible_states)) = self.stack.pop() {
            let old = self.state.array[index as usize];
            self.state.array[index as usize] &= remaining_possible_states;
            let new = self.state.array[index as usize];

            if old == new {
                continue;
            }

            if old.count_ones() > 1 {
                let _val = self.state.entropy_to_indices.remove(
                    Reverse(E::calculate::<_, BITS>(&self.probabilities, old)),
                    &index,
                );
                debug_assert!(_val);
            }

            if new == Wave::zero() {
                any_contradictions = true;
                continue;
            }

            if new.count_ones() > 1 {
                let _val = self.state.entropy_to_indices.insert(
                    Reverse(E::calculate::<_, BITS>(&self.probabilities, new)),
                    index,
                );
                debug_assert!(_val);
            }

            let new_tiles = tile_list_from_wave::<_, BITS>(new);

            for axis in Axis::ALL {
                let (mut x, mut y, mut z) = (
                    index % self.width(),
                    (index / self.width()) % self.height(),
                    index / self.width() / self.height(),
                );
                match axis {
                    Axis::X if x < self.width() - 1 => x += 1,
                    Axis::Y if y < self.height() - 1 => y += 1,
                    Axis::Z if z < self.depth() - 1 => z += 1,
                    Axis::NegX if x > 0 => x -= 1,
                    Axis::NegY if y > 0 => y -= 1,
                    Axis::NegZ if z > 0 => z -= 1,
                    _ => continue,
                };

                let index = x + y * self.width() + z * self.width() * self.height();

                let mut valid = Wave::zero();

                for &tile in new_tiles.iter() {
                    valid |= self.tiles[tile as usize].connections[axis as usize];
                }

                self.stack.push((index, valid));
            }
        }

        any_contradictions
    }

    #[inline]
    pub fn values(&self) -> Vec<u8> {
        let mut values = vec![0; self.state.array.len()];
        self.set_values(&mut values);
        values
    }

    #[inline]
    pub fn set_values(&self, values: &mut [u8]) {
        self.state
            .array
            .iter()
            .zip(values)
            .for_each(|(wave, value)| {
                *value = if wave.count_ones() == 1 {
                    wave.trailing_zeros() as u8
                } else {
                    u8::MAX
                }
            });
    }

    #[cfg(test)]
    fn all_collapsed(&self) -> bool {
        self.state
            .array
            .iter()
            .all(|&value| value.count_ones() == 1)
    }
}

fn find_any_with_early_stop<
    T,
    O: Send,
    I: IntoParallelIterator<Item = T>,
    F: Sync + Fn(T, &AtomicBool) -> Option<O>,
>(
    iterator: I,
    func: F,
) -> Option<O> {
    let stop_flag = AtomicBool::new(false);
    iterator.into_par_iter().find_map_any(|item| {
        func(item, &stop_flag).inspect(|_| stop_flag.store(true, Ordering::Relaxed))
    })
}

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

    let mut wfc = tileset.into_wfc::<ShannonEntropy>((100, 1000, 1));

    assert!(!wfc.all_collapsed());
    assert!(!wfc.collapse_all(&mut rng));
    assert!(
        wfc.all_collapsed(),
        "failed to collapse: {:?}",
        &wfc.state
            .array
            .iter()
            .map(|v| v.count_ones())
            .collect::<Vec<_>>()
    );
}

#[test]
fn initial_state() {
    let mut rng = SmallRng::from_entropy();

    let mut tileset = Tileset::<u8, 8>::default();
    let sea = tileset.add(1.0);
    let beach = tileset.add(1.0);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis::ALL);
    tileset.connect(sea, beach, &Axis::ALL);
    tileset.connect(beach, grass, &Axis::ALL);
    tileset.connect(grass, grass, &Axis::ALL);

    let mut state = vec![1 << sea | 1 << beach | 1 << grass; 9];
    assert_eq!(
        tileset
            .create_wfc_with_initial_state::<LinearEntropy>((3, 3, 1), state.clone())
            .state
            .array,
        state
    );
    state[4] = 1 << sea;
    #[rustfmt::skip]
    let expected = [
        7,3,7,
        3,1,3,
        7,3,7
    ];
    let mut wfc = tileset.into_wfc_with_initial_state::<ShannonEntropy>((3, 3, 1), state);
    assert_eq!(wfc.state.array, expected);
    wfc.collapse_all(&mut rng);
    assert_ne!(wfc.state.array, expected);
    wfc.reset();
    assert_eq!(wfc.state.array, expected);
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

    let mut wfc = tileset.into_wfc::<ShannonEntropy>((50, 50, 50));

    assert!(!wfc.all_collapsed());
    assert!(!wfc.collapse_all(&mut rng));
    assert!(
        wfc.all_collapsed(),
        "{:?}",
        &wfc.state
            .array
            .iter()
            .map(|v| v.count_ones())
            .collect::<Vec<_>>()
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

    let mut wfc = tileset.into_wfc::<ShannonEntropy>((5, 5, 5));

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
        let mut wfc = tileset.create_wfc::<ShannonEntropy>((10, 10, 1));

        assert!(!wfc.all_collapsed());

        if wfc.collapse_all(&mut rng) {
            assert!(!wfc.all_collapsed());
            // Make sure that at least one state has collapsed properly (aka that the error hasn't spread).
            assert!(wfc.state.array.iter().any(|&v| v.count_ones() == 1));
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
        .into_wfc::<ShannonEntropy>((10, 10, 10))
        .collapse_all_reset_on_contradiction(&mut rng);
}
