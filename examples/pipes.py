from markov import *

dim = 128

tileset = TaggingTileset()

empty = tileset.add(0.0, {"x": "no", "negx": "no", "y": "no", "negy": "no"})

straight_h, straight_v = tileset.add_mul(
    1.0, 2, {"x": "line", "negx": "line", "y": "no", "negy": "no"}
)

edge_dr, edge_dl, edge_ul, edge_ur = tileset.add_mul(
    1.0, 4, {"y": "line", "negy": "no", "x": "line", "negx": "no"}
)


end_r, end_d, end_l, end_u = tileset.add_mul(
    0.01, 4, {"y": "no", "negy": "no", "x": "line", "negx": "no"}
)

# Disallow self connections by specifying different self and connect-to tags.
cross = tileset.add(
    2.5,
    {
        "x": Tags(incoming="cross", outgoing="line"),
    },
    symmetry="X",
)

# tileset.connect_all()

wfc = tileset.tileset.create_wfc((dim, dim, 1))
tiles = np.zeros((wfc.num_tiles(), 3, 3), dtype=np.uint8)

tiles[straight_h][1] = 1
tiles[straight_v][:, 1] = 1

tiles[cross][1] = 1
tiles[cross][:, 1] = 1

tiles[edge_ur][1, 1:] = 1
tiles[edge_ur][:2, 1] = 1

tiles[edge_ul][1, :2] = 1
tiles[edge_ul][:1, 1] = 1

tiles[edge_dr][1, 1:] = 1
tiles[edge_dr][1:, 1] = 1

tiles[edge_dl][1, :1] = 1
tiles[edge_dl][1:, 1] = 1

arr = np.zeros((dim * 3, dim * 3), dtype=np.uint8)

for i in range(dim):
    wfc.collapse(i, empty)
    wfc.collapse(i * dim, empty)
    wfc.collapse(i * dim + (dim - 1), empty)
    wfc.collapse(i + (dim * (dim - 1)), empty)
    # wfc.collapse(i, empty)


writer = FfmpegWriter("out.avi", (dim * 3, dim * 3))

any_contradictions = collapse_all_with_callback(
    wfc, lambda: writer.write(map_2d(wfc.values()[0], arr, tiles)), skip=8
)
assert not any_contradictions
