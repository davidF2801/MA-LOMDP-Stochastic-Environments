# Understanding the Transition Probability Table

## What is this table?

The transition table is a **location-agnostic, general fire spreading model** learned from VIIRS satellite data. It does NOT store lat/lon coordinates. Instead, it learns:

> **"Given a local 3x3 pattern of fire states, what's the probability that the center cell will be on fire in the next time step?"**

## How it works:

### Step 1: Discretize the world
- The Earth is divided into a **30×60 grid** (30 latitude cells × 60 longitude cells)
- Each VIIRS fire detection is assigned to the nearest grid cell based on its lat/lon

### Step 2: Build time-binned fire grids
- Time is divided into bins (e.g., 1-minute bins, 12-hour bins, etc.)
- For each time bin, create a binary grid: `1` = fire detected, `0` = no fire

### Step 3: Learn local patterns
For each cell at each time step:
1. Look at the **3×3 neighborhood** (center cell + 8 neighbors)
2. Encode this pattern as a 9-bit number (0-511)
   - Bit 0 (most significant): center cell state (0 or 1)
   - Bits 1-8: neighbor states in order: top-left, top, top-right, left, right, bottom-left, bottom, bottom-right
3. Check what happened in the **next time step**: did the center cell catch fire?
4. Count occurrences: how many times did this pattern lead to fire?

### Step 4: Compute probabilities
For each of the 512 possible patterns:
- `total_count`: How many times we saw this pattern
- `fire_count`: How many times this pattern led to fire in the next step
- `p_fire = (fire_count + smoothing) / (total_count + 2*smoothing)`: Transition probability

## What the CSV columns mean:

| Column | Meaning |
|--------|--------|
| `pattern_index` | Unique ID (0-511) for each possible 3×3 pattern |
| `pattern_binary` | 9-bit binary string: `center + 8 neighbors` |
| `pattern_grid` | Visual 3×3 grid representation (center in middle) |
| `center_state` | State of center cell (0=no fire, 1=fire) |
| `active_neighbors` | Number of neighbors that are on fire (0-8) |
| `total_count` | How many times this pattern appeared in the data |
| `fire_count` | How many times this pattern led to fire in next step |
| `p_fire` | Probability of fire transition: P(fire at t+1 \| pattern at t) |

## Example:

**Pattern Index 1:**
- Binary: `000000001` = center=0, neighbors=[0,0,0,0,0,0,0,1]
- Grid:
  ```
  0 0 0
  0 0 0    ← center is 0 (no fire)
  0 0 1    ← bottom-right neighbor is 1 (on fire)
  ```
- Meaning: "Cell with no fire, but one neighbor (bottom-right) is on fire"
- `p_fire`: Probability this cell catches fire in the next time step

## Why NOT location-specific?

The model assumes **fire spreading behavior is similar everywhere** - it's a general rule learned from all locations. This makes sense because:
- Fire spreads based on local conditions (neighbors, fuel, weather)
- The same local pattern should behave similarly regardless of lat/lon
- We can use this model to simulate fire anywhere on the grid

## How to use it:

1. **For simulation**: At each grid cell, look at its 3×3 neighborhood, encode it as a pattern index, look up `p_fire`, and randomly decide if fire spreads based on that probability.

2. **For analysis**: Patterns with high `p_fire` are "dangerous" - they lead to fire spread. Patterns with low `p_fire` are "safe".

## Note about the data:

If you see `p_fire = 0.5` for many patterns with `total_count = 0`, this means:
- These patterns were never observed in the training data
- The probability defaults to 0.5 due to Laplace smoothing (prior belief)
- In practice, these patterns will use the global average fire rate

