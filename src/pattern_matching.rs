use crate::arrays::{Array3D, ArrayPair};
use crate::{bespoke_regex, WILDCARD};

pub struct OverlappingRegexIter<'a> {
    regex: &'a bespoke_regex::BespokeRegex,
    haystack: &'a [u8],
    offset: usize,
}

impl<'a> OverlappingRegexIter<'a> {
    pub fn new(regex: &'a bespoke_regex::BespokeRegex, haystack: &'a [u8]) -> Self {
        Self {
            regex,
            haystack,
            offset: 0,
        }
    }
}

impl<'a> Iterator for OverlappingRegexIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self.regex.find(&self.haystack[self.offset..]) {
            Some(start) => {
                let index = self.offset + start;
                self.offset += start + 1;
                Some(index)
            }
            None => None,
        }
    }
}

pub struct Permutation {
    pub bespoke_regex: bespoke_regex::BespokeRegex,
    pattern_len: usize,
    pub to: Array3D,
}

impl Permutation {
    pub fn new(state: &Array3D<&mut [u8]>, pair: ArrayPair) -> Self {
        let mut bespoke_values = Vec::new();
        let mut pattern_len = 0;

        let row_offset = state.width() - pair.from.width();

        let layer_offset = (state.width() * (state.height() - pair.from.height())) + row_offset;
        for (z, layer) in pair.from.layers().enumerate() {
            for (y, row) in layer.chunks_exact(pair.from.width()).enumerate() {
                for &value in row {
                    if value == WILDCARD {
                        bespoke_values.push(bespoke_regex::LiteralsOrWildcards::Wildcards(1));
                    } else {
                        bespoke_values
                            .push(bespoke_regex::LiteralsOrWildcards::Literal(vec![value]));
                    }
                }

                pattern_len += pair.from.width();

                if y < pair.from.height() - 1 {
                    bespoke_values.push(bespoke_regex::LiteralsOrWildcards::Wildcards(row_offset));
                    pattern_len += row_offset;
                }
            }

            if z < pair.from.depth() - 1 {
                bespoke_values.push(bespoke_regex::LiteralsOrWildcards::Wildcards(layer_offset));
                pattern_len += layer_offset;
            }
        }

        Self {
            pattern_len,
            bespoke_regex: bespoke_regex::BespokeRegex::new(&bespoke_values),
            to: pair.to,
        }
    }

    pub fn width(&self) -> usize {
        self.to.width()
    }

    pub fn height(&self) -> usize {
        self.to.height()
    }

    pub fn depth(&self) -> usize {
        self.to.depth()
    }

    #[inline]
    pub fn coords(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        (0..self.depth()).flat_map(move |z| {
            (0..self.height()).flat_map(move |y| (0..self.width()).map(move |x| (x, y, z)))
        })
    }
}

pub fn match_pattern(regex: &Permutation, state: &Array3D<&mut [u8]>, index: u32) -> bool {
    let end = index as usize + regex.pattern_len;

    if end > state.inner.len()
        || !state.shape_is_inbounds(
            index as _,
            regex.to.width(),
            regex.to.height(),
            regex.to.depth(),
        )
    {
        return false;
    }
    regex
        .bespoke_regex
        .is_immediate_match(&state.inner[index as usize..end])
}
