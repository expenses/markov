from markov import *

axis = ["x", "y", "negx", "negy"]


VARIANTS = {"X": ["x"], "I": ["x", "y"], "L": axis, "T": axis, "": axis}

OUTGOING = {
    (0, "X"): axis,
    (0, "I"): ["x", "negx"],
    (1, "I"): ["y", "negy"],
}


def mirror_diag(dir):
    pass


def rots_for_sym(dir, sym):
    ret = [dir]
    if sym == "X":
        return axis
    if sym == "I":
        ret.append(FLIPPED[dir])
    if sym == "L":
        ret.append(MIRRORED[dir])
    if sym == "T":
        if dir == "x" or dir == "negx":
            ret.append(FLIPPED[dir])
    return ret

NUM_DIRECTIONS = {
    "X": [[0,1,2,3]],
    "I": [[0,2],[1,3]],
}

class Tile:
    def __init__(self, wfc, name, symmetry="", probability=1.0):
        self.symmetry = symmetry.upper()
        variants = VARIANTS[self.symmetry]
        self.probability = probability / len(variants)
        self.name = name

        self.tiles = dict((i, wfc.add(self.probability)) for i in range(len(variants)))

        print(self.tiles)

    def get(self, i):
        return self.tiles[i % len(self.tiles)]

    def connect_into(self, left_variant, direction, right_variant):
        right_variant = self.tiles[right_variant % len(self.tiles)]
        wfc.connect(left_variant, right_variant, [direction])
        print(left_variant, right_variant, [direction])



    def connect(self, rot_i, right, right_rot_i):
        for (variant_index, variant_directions) in enumerate(NUM_DIRECTIONS[self.symmetry]):
            for outgoing_direction in variant_directions:
                outgoing_direction = (outgoing_direction + rot_i) % 4
                variant = self.tiles[variant_index]

                incoming_variant = right.get((outgoing_direction+2+right_rot_i))

                print(f"{self.name} ({self.get(variant_index)}) is going out on {axis[outgoing_direction]} to {right.name} ({incoming_variant})")

                wfc.connect(self.get(variant_index), incoming_variant, [axis[outgoing_direction]])
               #for right_variant in range(len(right.tiles)):
               #    print(variant, outgoing_direction, right.tiles[right_variant])


        '''
        for i in range(4):
            variant = i % len(self.tiles)
            #print(i // (variant+1))
            #right_variant = (vvv + (right_rot_i-rot_i))
            #print(right_variant)
            #for direction in range(i // (variant+1)):
            #    direction = (direction + rot_i) % 4
            #    right_variant = direction+(right_rot_i-rot_i)
            #    print(f"{self.name} {variant} -> {axis[direction]} -> {right.name} {right_variant}")
            #    wfc.connect(self.get(variant), right.get(direction+(right_rot_i-rot_i)), [axis[direction]])
        '''
        '''
        for set in NUM_DIRECTIONS[self.symmetry]:
            for i in set:
                left_variant = (i + rot_i)
                right_variant = (i + right_rot_i)
                #print(self.get(i + rot_i), right.get(i + right_rot_i), axis[i])
        '''
        '''
        num_connections = len(self.tiles) * len(right.tiles)
        print(num_connections)

        for i in range(max(len(self.tiles), 0)):
            for j, o in enumerate(OUTGOING[((i % len(self.tiles)), self.symmetry)]):
                print(
                    self.tiles[(rot_i + i) % len(self.tiles)],
                    right.tiles[(right_rot_i + i) % len(right.tiles)],
                    o,
                )
        '''
        print("!!")



class Tileset:
    def __init__(self):
        self.tiles = []

    def add(self, tile):
        index = len(self.tiles)
        self.tiles.append(tile)
        return index

    def connect(self, left, *rights, rot=0):
        for right in rights:
            if type(right) is not tuple:
                right = (right, 0)
            right, right_rot = right
            self.tiles[left].connect(rot, self.tiles[right], right_rot)


tileset = Tileset()
wfc = Wfc((100, 100, 1))
empty = tileset.add(Tile(wfc, "empty", symmetry="X", probability=0.0))
line = tileset.add(Tile(wfc, "line", symmetry="I"))
cross = tileset.add(Tile(wfc, "cross", symmetry="X"))

tileset.connect(empty, empty)

tileset.connect(empty,(line,1))


tileset.connect(line, line)
tileset.connect(line, (line, 1), rot=1)

tileset.connect(cross, cross)
tileset.connect(cross, cross, rot=1)

tileset.connect(cross, (cross,1), rot=1)




tileset.connect(cross, cross)
tileset.connect(cross, cross, rot=1)
tileset.connect(cross, (cross,1), rot=1)

tileset.connect(cross, line)
tileset.connect(cross, line, rot=1)


wfc.setup_state()
wfc.collapse_all()
print(wfc.all_collapsed())

palette = Palette(PICO8_PALETTE.srgb + [(128, 128, 128)] * 100)
save_image("out.png", wfc.values()[0],palette=palette)

'''

tileset.connect(line, (line, 1), rot=1)

tileset.connect(cross, cross)
tileset.connect(cross, cross, rot=1)

tileset.connect(cross, line)

tileset.connect(cross, line, rot=1)

tileset.connect(empty, (line, 1))
'''
