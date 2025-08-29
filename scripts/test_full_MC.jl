using LinearAlgebra
using Random
using Printf
using StatsBase
using Infiltrator

# Grid size
rows, cols = 3, 3
K = rows * cols
S = 2
N = S^K

println("Number of states: $N")

# --- Encode/decode states as integers <-> binary vectors ---
function int_to_state(idx::Int, K::Int)
    binary_digits = reverse(digits(idx-1, base=2, pad=K))
    return binary_digits
end

function state_to_int(state::Vector{Int})
    val = 0
    for s in state
        val = val*2 + s
    end
    return val+1
end

# --- Grid neighbor map (4-neighbors) ---
function build_neighbors(rows, cols)
    neigh = Dict{Int, Vector{Int}}()
    for r in 1:rows, c in 1:cols
        j = (r-1)*cols + c
        neighs = Int[]
        if r > 1; push!(neighs, (r-2)*cols + c); end
        if r < rows; push!(neighs, r*cols + c); end
        if c > 1; push!(neighs, (r-1)*cols + c - 1); end
        if c < cols; push!(neighs, (r-1)*cols + c + 1); end
        neigh[j] = neighs
    end
    return neigh
end

neighbors = build_neighbors(rows, cols)

# --- Transition for one cell ---
function cell_transition(xj, neighvals)
    if xj == 1
        return [0.1, 0.9]  # stays 1 with 0.9
    else
        if any(neighvals .== 1)
            return [0.7, 0.3]  # ignite with prob 0.3
        else
            return [1.0, 0.0]  # stay 0
        end
    end
end

# --- Build transition matrix T ---
@time begin
    T = zeros(Float64, N, N)
    for x_idx in 1:N
        x = int_to_state(x_idx, K)
        for xprime_idx in 1:N
            xprime = int_to_state(xprime_idx, K)
            prob = 1.0
            for j in 1:K
                neighvals = x[neighbors[j]]
                pj = cell_transition(x[j], neighvals)
                prob *= pj[xprime[j]+1]
            end
            T[x_idx, xprime_idx] = prob
        end
    end
end
println("Transition matrix built: size = $(size(T))")

# --- Simulation ---
steps = 10
state = zeros(Int, K)  # start all 0
state[5] = 1           # ignite center cell

function print_grid(state, rows, cols)
    for r in 1:rows
        for c in 1:cols
            print(state[(r-1)*cols + c], " ")
        end
        println()
    end
    println()
end

println("Initial state:")
print_grid(state, rows, cols)
@infiltrate
for t in 1:steps
    global state
    x_idx = state_to_int(state)
    probs = T[x_idx, :]
    new_idx = sample(1:N, Weights(probs))
    state = int_to_state(new_idx, K)
    println("Step $t:")
    print_grid(state, rows, cols)
end
