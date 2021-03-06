module JSom

using DataFrames
using DataStructures
using Distances
using Hexagons
using StatsBase

import Base.size


export
    GridSOM,
    HexSOM,

    # SOM objects methods
    get_unit_weight,
    set_unit_weight,
    size,
    update,
    get_BMU,
    activate,
    sequential_random_epoch,
    sequential_epoch,
    quantize,
    quantization_error,
    bmu_map,
    activation_response,
    umatrix,
    reset_state,

    # Neighborhood functions
    Gaussian_Neighborhood,
    Ricker_Neighborhood,
    Triangular_Neighborhood
    


abstract SOM
abstract NeighborhoodFunction
type Gaussian_Neighborhood      <: NeighborhoodFunction end
type Ricker_Neighborhood        <: NeighborhoodFunction end
type Triangular_Neighborhood    <: NeighborhoodFunction end


function SOM__init__(this::SOM, x::Int, y::Int, input_len::Int;
                     σ::Float64=1.0, η::Float64=0.5, seed=0)
    this.t = 0
    this.epoch = 0
    this.λ = 0
    this.σ₀ = σ
    this.η₀ = η
    this.weights = Array{Float64}((x,y,input_len))
    this.activation_map = Array{Float64}((x,y))
    this.τ = τ_inverse
    this.ħ = Gaussian_Neighborhood()
    this.dist = euclidean
    this.seed = seed != 0 ? seed : round(Int, rand()*1e7)
    this.rng = MersenneTwister(seed)
    init_weights(this)
end


type GridSOM <: SOM
    weights::Array{Float64,3}
    activation_map::Array{Float64,2}
    η₀::Float64
    σ₀::Float64
    λ::Float64
    t::Int
    epoch::Int
    τ::Function
    ħ::NeighborhoodFunction
    dist::Function
    seed::Int
    rng::MersenneTwister

    function GridSOM(x::Int, y::Int, input_len::Int;
                 σ::Float64=1.0, η::Float64=0.5, seed=0)
        this = new()
        SOM__init__(this, x, y, input_len, σ=σ, η=η, seed=seed)
        return this
    end
end


type HexSOM <: SOM
    weights::Array{Float64,3}
    activation_map::Array{Float64,2}
    η₀::Float64
    σ₀::Float64
    λ::Float64
    t::Int
    epoch::Int
    τ::Function
    ħ::NeighborhoodFunction
    dist::Function
    seed::Int
    rng::MersenneTwister

    function HexSOM(x::Int, y::Int, input_len::Int;
                    σ::Float64=1.0, η::Float64=0.5, seed=0)
        this = new()
        SOM__init__(this, x, y, input_len, σ=σ, η=η, seed=seed)
        return this
    end
end


function size(som::SOM)
    return size(som.activation_map)
end


# --------------------------
# Neighboring units

function neighbor_units(som::GridSOM, x::Int, y::Int)
    r, c = size(som)
    neighbors = [
        (x-1,y-1), (x-1,y), (x-1,y+1),
         (x,y-1),            (x,y+1),
        (x+1,y-1), (x+1,y), (x+1,y+1)
    ]
    return neighbor_units(neighbors, r, c)
end

function neighbor_units(som::HexSOM, x::Int, y::Int)
    r, c = size(som)
    neighbors = [(n.x, n.y) for n in Hexagons.neighbors(HexagonAxial(x, y))]
    return neighbor_units(neighbors, r, c)
end

function neighbor_units(neighbors::Vector{Tuple{Int,Int}}, r, c)
    f(u) = 0 < u[1] ≤ r && 0 < u[2] ≤ c
    filter(f, neighbors)
end

neighbor_units(som::SOM, u::Tuple{Int,Int}) = neighbor_units(som, u...)


function hexagonal(u::Tuple{Int,Int}, v::Tuple{Int,Int})
    return float(Hexagons.distance(HexagonAxial(u...), HexagonAxial(v...)))
end


# --------------------------
# Decay functions

function τ_inverse(x::Float64, t::Int, λ::Float64)
    return x / (1 + t/λ)
end


function τ_exponential(x::Float64, t::Int, λ::Float64)
    return x * exp(-t/λ)
end


# --------------------------
# Neighborhood functions

function ħ(som::SOM, u::Tuple{Int,Int}, bmu::Tuple{Int,Int}, σ::Float64)
    return _ħ(som, som.ħ, u, bmu, σ)
end

# Gaussian
_ħ(som::SOM, ::Gaussian_Neighborhood, u, v, σ) = ħ_gaussian(som, u, v, σ)
ħ_gaussian(::GridSOM, u, v, σ) = ħ_gaussian(σ, euclidean(collect(u), collect(v)))
ħ_gaussian(::HexSOM, u, v, σ) = ħ_gaussian(σ, hexagonal(u, v))
ħ_gaussian(σ::Float64, d::Float64) = exp(-d^2 / (2 * σ^2))


# Mexican hat
_ħ(som::SOM, ::Ricker_Neighborhood, u, v, σ) = ħ_ricker(som, u, v, σ)
ħ_ricker(::GridSOM, u, v, σ) = ħ_ricker(σ, euclidean(collect(u), collect(v)))
ħ_ricker(::HexSOM, u, v, σ) = ħ_ricker(σ, hexagonal(u, v))
ħ_ricker(σ::Float64, d::Float64) = (1 - d^2 / σ^2) * ħ_gaussian(σ, d)


# Triangular
_ħ(som::SOM, ::Triangular_Neighborhood, u, v, σ) = ħ_triangular(som, u, v, σ)
ħ_triangular(::GridSOM, u, v, σ) = ħ_triangular(σ, euclidean(collect(u), collect(v)))
ħ_triangular(::HexSOM, u, v, σ) = ħ_triangular(σ, hexagonal(u, v))
ħ_triangular(σ::Float64, d::Float64) = abs(d) ≤ σ ? 1 - abs(d) / σ : 0.0


# --------------------------
# Getter / Setters

function get_unit_weight(som::SOM, i::Int, j::Int)
    return som.weights[i,j,:]
end

function get_unit_weight(som::SOM, c::Tuple{Int, Int})
    return get_unit_weight(som, c[1], c[2])
end

function set_unit_weight(som::SOM, i::Int, j::Int, w::Array{Float64})
    som.weights[i,j,:] = w
end

function set_unit_weight(som::SOM, i::Int, j::Int, w::Float64)
    som.weights[i,j,:] = [w]
end

function set_unit_weight(som::SOM, c::Tuple{Int, Int}, w)
    set_unit_weight(som, c[1], c[2], w)
end


# --------------------------
# Main API

function init_weights(som::SOM)
    l = size(som.weights, 3)
    for k in eachindex(som.activation_map)
        i, j = ind2sub(som.activation_map, k)
        w = rand(som.rng, l) * 2 - 1
        w /= norm(w)
        set_unit_weight(som, i, j, vec(w))
    end
end


function init_λ(som::SOM, num_iter::Int)
    som.λ = (som.t + num_iter) / 2
end


function update(som::SOM, input::Array, bmu::Tuple)
    som.t += 1
    input = vec(input)
    ηₜ = som.τ(som.η₀, som.t, som.λ)
    σₜ = som.τ(som.σ₀, som.t, som.λ)
    for k in eachindex(som.activation_map)
        u = ind2sub(som.activation_map, k)
        h = ħ(som, u, bmu, σₜ)
        weight = vec(get_unit_weight(som, u))
        weight = weight + h * ηₜ * (input - weight)
        set_unit_weight(som, u, weight)
    end
end


function get_BMU(som::SOM, input::Array)
    activate(som, input)
    return ind2sub(size(som), indmin(som.activation_map))
end


function activate(som::SOM, input::Array)
    input = reshape(input, (1,1,size(input, 2)))
    for k in eachindex(som.activation_map)
        i, j = ind2sub(som.activation_map, k)
        weight = get_unit_weight(som, i, j)
        som.activation_map[k] = som.dist(vec(input), vec(weight))
    end
end


function quantize(som::SOM, data::Array)
    q = similar(data)
    for i=1:size(data, 1)
        input = data[i,:]
        bmu = get_BMU(som, input)
        q[i,:] = get_unit_weight(som, bmu)
    end
    return q
end


function activation_response(som::SOM, data::Array)
    a = zeros(size(som))
    for i=1:size(data, 1)
        input = data[i,:]
        bmu = get_BMU(som, input)
        a[bmu...] += 1
    end
    return a
end


function umatrix(som::SOM)
    u_matrix = similar(som.activation_map)
    l = size(som)
    indices = [(x,y) for x in 1:l[1], y in 1:l[2]]
    for u in indices
        u_matrix[u...] = 0.0
        neighbors = neighbor_units(som, u)
        w = get_unit_weight(som, u)
        for n in neighbors
            v = get_unit_weight(som, n)
            u_matrix[u...] += som.dist(collect(v), collect(w))
        end
        u_matrix[u...] /= length(neighbors)
    end
    return u_matrix
end


function bmu_map(som::SOM, data::Array)
    bmus = DefaultDict(Tuple{Int,Int}, Vector{Vector}, Vector{Vector})
    for i=1:size(data, 1)
        input = data[i,:]
        bmu = get_BMU(som, input)
        push!(bmus[bmu...], vec(input))
    end
    return bmus
end


function sequential_random_epoch(som::SOM, data::Array, num_iter::Int)
    som.epoch += 1
    init_λ(som, num_iter)
    for t = 0:num_iter
        i = rand(som.rng, 1:size(data, 1))
        input = data[i,:]
        update(som, input, get_BMU(som, input))
    end
end


function sequential_epoch(som::SOM, data::Array, epochs=1)
    for i=1:epochs
        som.epoch += 1
        num_iter = size(data, 1)
        init_λ(som, num_iter)
        for t in shuffle(som.rng, collect(1:num_iter))
            input = data[t,:]
            update(som, input, get_BMU(som, input))
        end
    end
end


function quantization_error(som::SOM, data::Array)
    err = 0
    for i=1:size(data, 1)
        input = data[i,:]
        weight = get_unit_weight(som, get_BMU(som, input))
        err += som.dist(vec(weight), vec(input))
    end
    return err / size(data, 1)
end


function reset_state(som::SOM)
    som.t = 0
    som.epoch = 0
    som.rng = MersenneTwister(som.seed)
    init_weights(som)
end


end
