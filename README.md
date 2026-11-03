# ⚡ ReLU-AStar: Exact A* Search as a Recurrent Neural Circuit

> **Canonical, optimal A* pathfinding compiled strictly into fixed linear maps, bias additions, and ReLUs. No training. No queues. No branching.**

---

### 🧠 What is this?

This repository contains an analytical, closed-form construction of the **A\* graph search algorithm** implemented entirely as an artificial recurrent neural network (RNN). 

Unlike typical "Neural A*" projects in machine learning, this is **not an approximation** and uses **no gradient descent**. It is an exact, bit-for-bit, 1-to-1 algorithmic isomorphism with canonical A* executed purely through tensor arithmetic.

### 🚫 The "No-CPU" Guarantees (Inside the Engine)

The recurrent inference step (`forward()`) contains:
- ❌ **No `if` / `else` / conditional branches**
- ❌ **No `heapq` / priority queues**
- ❌ **No `argmin` / `min` operators**
- ❌ **No variable loop bounds (static $N$ ticks)**
- ❌ **No dynamic memory allocations**

Everything runs exclusively through matrix products (`@`), broadcast differences, and $\text{ReLU}(x) = \max(0, x)$.

### 🔬 Core Neural Primitives Used

1. **Ramp-Difference Step Comparator:** Simulates discrete Boolean logic ($x < y$) using two opposing ReLU ramps: $\text{ReLU}(d) - \text{ReLU}(d - 1)$.
2. **Pairwise Tournament Argmin:** Replaces the priority heap with an $N \times N$ matrix broadcast, computing an $O(1)$ Winner-Take-All pop via tensor summation.
3. **Lexicographic Radix Packing:** Encodes $(f, h, \text{index})$ tie-breaking into a single scalar guaranteed to stay within IEEE 754 `float64` mantissa precision ($< 2^{52}$).
4. **Analog ReLU Multiplexers:** Implements conditional variable updates ($a \ ? \ x : y$) using gated linear threshold arithmetic.
5. **Arithmetic Latches:** Freezes circuit state upon reaching the goal using an arithmetic inhibitor without breaking the loop.

### 🚀 Target Use-Cases

- **Neuromorphic & Photonic Computing:** Can execute on analog crossbars or optical tensor processors that lack standard Von Neumann CPUs.
- **Hard Real-Time Avionics:** 100% deterministic, constant-time execution with zero branch-prediction penalties.
- **Massively Parallel Search:** Run tens of thousands of A* pathfinding instances simultaneously on GPU tensor cores without CPU-GPU synchronization.
