# Passing Assembly Test Visuals

Color key: `G`=green `B`=blue `Y`=yellow `R`=red

---

## 1. pass_single_block.json
**Single block on ground** — simplest valid assembly

```
Top (xy)        Front (xz)      Side (yz)

y               z               z
2  . . .        2  . . .        2  . . .
1  . G .        1  . G .        1  . G .
0  . . .        0  -----        0  -----
   0 1 2  x        0 1 2  x        0 1 2  y
```

---

## 2. pass_tower.json
**Vertical tower of 3 blocks** — tests vertical support chain

```
Top (xy)        Front (xz)      Side (yz)

y               z               z
2  . . .        3  . R .        3  . R .
1  . * .        2  . B .        2  . B .
0  . . .        1  . G .        1  . G .
   0 1 2  x     0  -----        0  -----
                   0 1 2  x        0 1 2  y

* = stack of G, B, R
```

---

## 3. pass_l_shape.json
**L-shape on ground** — tests horizontal adjacency

```
Top (xy)        Front (xz)      Side (yz)

y               z               z
2  . . .        2  . . .        2  . . .
1  . Y .        1  Y Y .        1  Y Y .
0  Y Y .        0  -----        0  -----
   0 1 2  x        0 1 2  x        0 1 2  y
```

---

## 4. pass_cantilever_at_limit.json
**Column + 2-block arm** — cantilever exactly at limit (dist=2)

```
Top (xy)            Front (xz)          Side (yz)

y                   z                   z
2  . . . .          2  G B B .          2  . G .
1  G B B .          1  G . . .          1  . G .
0  . . . .          0  -----            0  -----
   0 1 2 3  x          0 1 2 3  x          0 1 2  y

Cantilever dist:    0  1 2
```

---

## 5. pass_bridge.json
**Two pillars, span built from both sides inward** — cantilever resets at second pillar

Build order: pillars first, then span from both ends toward center.

```
Top (xy)                        Front (xz)

y                               z
2  . . . . . .                  2  G B B Y Y G
1  G B B Y Y G                  1  G . . . . G
0  . . . . . .                  0  -----
   0 1 2 3 4 5  x                  0 1 2 3 4 5  x

Side (yz)

z
2  . G .
1  . G .
0  -----
   0 1 2  y

Cantilever dist:  0 1 2 2 1 0
```

---

## 6. pass_staircase.json
**Cantilevered staircase** — 2 horizontal hops from ground (at limit)

Tests that vertical moves are free in cantilever calculation.

```
Top (xy)            Front (xz)              Side (yz)

y                   z                       z
2  . . . .          3  . B Y .              3  . * .
1  G B Y .          2  G B . .              2  . * .
0  . . . .          1  G . . .              1  . * .
   0 1 2 3  x       0  -----                0  -----
                       0 1 2 3  x               0 1 2  y

Cantilever dist:    0  1 2                  * = multiple blocks at y=1
```
