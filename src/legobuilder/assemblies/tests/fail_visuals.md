# Failing Assembly Test Visuals

Color key: `G`=green `B`=blue `Y`=yellow `R`=red
Error marker: `!` suffix (e.g. `B!` = block that causes an error)

---

## 1. fail_out_of_bounds.json
**Block at x=5 in a 5-wide grid** — grid bounds check

```
Top (xy)                 Front (xz)

y                        z
4  . . . . . |           4  . . . . . |
3  . . . . . |           3  . . . . . |
2  . . G . . | B!        2  . . . . . |
1  . . . . . |           1  . . G . . | B!
0  . . . . . |           0  -----------
   0 1 2 3 4  [5]  x        0 1 2 3 4  [5]  x

Side (yz)                | = grid boundary
                         [5] = out of bounds
z
4  . . . . .
3  . . . . .
2  . . G . .
1  . . . . .
0  ----------
   0 1 2 3 4  y
```

---

## 2. fail_z_zero.json
**Block at z=0 (below ground)** — z >= 1 requirement

```
Top (xy)        Front (xz)      Side (yz)

y               z               z
2  . . .        2  . . .        2  . . .
1  . G!.        1  . . .        1  . . .
0  . . .        0  . G!.        0  . G!.
   0 1 2  x     ==========      ==========
                   0 1 2  x        0 1 2  y

                ========== = ground level (z=0 is below)
```

---

## 3. fail_duplicate_position.json
**Two blocks at same (1,1,1)** — duplicate position check

```
Top (xy)        Front (xz)      Side (yz)

y               z               z
2  . . .        2  .  .  .      2  .  .  .
1  .G/Y!.       1  . G/Y!.      1  . G/Y!.
0  . . .        0  ------       0  ------
   0 1 2  x        0  1  2  x      0  1  2  y

G/Y! = green and yellow both claim (1,1,1)
```

---

## 4. fail_duplicate_id.json
**Two blocks with id=1** — duplicate ID check

```
Top (xy)        Front (xz)      Side (yz)

y               z               z
2  . . .        2  . . .        2  .
1  . . .        1  G B .        1  *
0  G B .        0  -----        0  --
   0 1 2  x        0 1 2  x       0  y

Both blocks have id=1     * = both at y=0
```

---

## 5. fail_invalid_color.json
**Block with color "purple"** — color validation

```
Top (xy)        Front (xz)      Side (yz)

y               z               z
2  . . .        2  . . .        2  . . .
1  . G .        1  .?! G        1  ?!G .
0  .?! .        0  -----        0  -----
   0 1 2  x        0 1 2  x        0 1 2  y

?! = "purple" block at (1,0,1) — unknown color
```

---

## 6. fail_floating.json
**Block floating in mid-air** — build order support (no neighbor)

```
Top (xy)              Front (xz)

y                     z
3  . . . R!           4  . . . R!
2  . . . .            3  . . . .
1  . . . .            2  . . . .
0  G . . .            1  G . . .
   0 1 2 3  x         0  --------
                          0 1 2 3  x

Side (yz)

z
4  . . . R!
3  . . . .
2  . . . .
1  G . . .
0  --------
   0 1 2 3  y

R! at (3,3,4) — no neighbors, not on ground
```

---

## 7. fail_build_order.json
**Valid structure, wrong JSON order** — build order check

The final structure is a 2-high tower, but block at z=2 is listed before z=1.

```
Top (xy)        Front (xz)              Side (yz)

y               z                       z
2  . . .        2  . B .  ← placed 1st  2  . B .
1  . * .        1  . G .  ← placed 2nd  1  . G .
0  . . .        0  -----                0  -----
   0 1 2  x        0 1 2  x                0 1 2  y

* = B on top of G       Error: B has no support when placed
```

---

## 8. fail_cantilever.json
**Column + 3-block arm** — exceeds cantilever limit of 2

```
Top (xy)                Front (xz)

y                       z
2  . . . . .            2  G B B R!.
1  G B B R!.            1  G .  . . .
0  . . . . .            0  ---------
   0 1 2 3 4  x            0 1  2 3 4  x

Side (yz)

z
2  . G .
1  . G .
0  -----
   0 1 2  y

Cantilever dist:  0 1 2 3!
                          ^ exceeds limit
```

---

## 9. fail_disconnected.json
**Two separate blocks on ground** — ground connectivity warning (passes, but warns)

```
Top (xy)                    Front (xz)

y                           z
4  . . . . R                4  . . . . .
3  . . . . .                3  . . . . .
2  . . . . .                2  . . . . .
1  . . . . .                1  G . . . R
0  G . . . .                0  ----------
   0 1 2 3 4  x                0 1 2 3 4  x

Side (yz)

z
4  . . . . .
3  . . . . .
2  . . . . .
1  G . . . R
0  ----------
   0 1 2 3 4  y

Warning: two islands, no 6-connected path between them
```

---

## 10. fail_staircase.json
**Cantilevered staircase, 3 horizontal hops** — exceeds cantilever limit

Same pattern as pass_staircase but with one more step.

```
Top (xy)                Front (xz)

y                       z
2  . . . . .            4  . . Y R!.
1  G B Y R!.            3  . B Y .  .
0  . . . . .            2  G B .  .  .
   0 1 2 3 4  x         1  G .  .  .  .
                         0  -----------
                            0 1  2  3  4  x

Side (yz)

z
4  . * .
3  . * .
2  . * .
1  . * .
0  -----
   0 1 2  y

Cantilever dist:  0 1 2 3!
                          ^ exceeds limit

* = all blocks at y=1
```
