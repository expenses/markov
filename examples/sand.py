# coding: utf-8
from util import *
import sys
dim = int(sys.argv[1])
arr = np.zeros((dim, dim), dtype=np.uint8)
ffmpeg = FfmpegOutput("out.avi", width=dim, height=dim, skip=40)


def sand(c):
    return (
        PatternWithOptions(
            f"0={c}", apply_all=True, chance=0.1, node_settings=NodeSettings(count=1)
        ),
        Markov(
            One(
                PatternWithOptions(
                    f"{c}0,{c}0=00,{c}{c}",
                    allow_dimension_shuffling=False,
                    allow_flip=False,
                ),
                PatternWithOptions(
                    f"0{c},0{c}=00,{c}{c}",
                    allow_dimension_shuffling=False,
                    allow_flip=False,
                ),
            ),
            PatternWithOptions(
                f"{c},0=0,{c}", apply_all=True, allow_dimension_shuffling=False, allow_flip=False
            ),
        ),
    )


rep2(
    arr,
    Sequence(
        *(
            sand("R")
            + sand("O")
            + sand("Y")
            + sand("G")
            + sand("U")
            + sand("I")
            + sand("P")
        )
    ),
    ffmpeg=ffmpeg,
)
