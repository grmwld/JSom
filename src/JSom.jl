module JSom

using DataFrames
using DataStructures
using Distances
using StatsBase

import Base.size


export
    SOM,
    _τ_inverse,
    _τ_exponential,
    _ħ_gaussian,
    _ħ_ricker,
    _ħ_triangular,
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
    reset_state



type SOM
    weights::Array{Float64,3}
    activation_map::Array{Float64,2}
    η::Float64
    σ::Float64
    λ::Float64
    t::Int
    epoch::Int
    τ::Function
    ħ::Function
    dist::Function
    seed::Int
    rng::MersenneTwister

    function SOM(x::Int, y::Int, input_len::Int;
                 σ::Float64=1.0, η::Float64=0.5, seed=0)
        this = new()
        this.t = 0
        this.epoch = 0
        this.λ = 0
        this.σ = σ
        this.η = η
        this.weights = Array{Float64}((x,y,input_len))
        this.activation_map = Array{Float64}((x,y))
        this.τ = _τ_inverse
        this.ħ = _ħ_gaussian
        this.dist = euclidean
        this.seed = seed != 0 ? seed : round(Int, rand()*1e7)
        this.rng = MersenneTwister(seed)
        init_weights(this)
        return this
    end

end


function size(som::SOM)
    return size(som.activation_map)
end


# --------------------------------------
# Private methods
# --------------------------------------

function __neighbor_units(som::SOM, x::Int, y::Int)
    r, c = size(som)
    n = [(x-1,y-1), (x-1,y), (x-1,y+1),
          (x,y-1),            (x,y+1),
         (x+1,y-1), (x+1,y), (x+1,y+1)]
    f(u) = 0 < u[1] ≤ r && 0 < u[2] ≤ c
    return filter!(f, n)
end

function __neighbor_units(som::SOM, u::Tuple{Int,Int})
    return __neighbor_units(som, u...)
end


# --------------------------------------
# Protected methods
# --------------------------------------

# --------------------------
# Decay functions

function _τ_inverse(x::Float64, t::Int, λ::Float64)
    return x / (1 + t/λ)
end


function _τ_exponential(x::Float64, t::Int, λ::Float64)
    return x * exp(-t/λ)
end


# --------------------------
# Neighborhood functions

function _ħ_gaussian(u::Tuple{Int,Int}, bmu::Tuple{Int,Int}, σ::Float64)
    d = euclidean(collect(u), collect(bmu))
    return exp(-d^2 / (2 * σ^2))
end


function _ħ_ricker(u::Tuple{Int,Int}, bmu::Tuple{Int,Int}, σ::Float64)
    d = euclidean(collect(u), collect(bmu))
    return (1 - d^2 / σ^2) * _ħ_gaussian(u, bmu, σ)
end


function _ħ_triangular(u::Tuple{Int,Int}, bmu::Tuple{Int,Int}, σ::Float64)
    d = euclidean(collect(u), collect(bmu))
    return abs(d) ≤ σ ? 1 - abs(d) / σ : 0.0
end


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
    η = som.τ(som.η, som.t, som.λ)
    σ = som.τ(som.σ, som.t, som.λ)
    for k in eachindex(som.activation_map)
        u = ind2sub(som.activation_map, k)
        ħ = som.ħ(u, bmu, σ)
        weight = vec(get_unit_weight(som, u))
        weight = weight + ħ * η * (input - weight)
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
    umatrix = similar(som.activation_map)
    l = size(som)
    indices = [(x,y) for x in 1:l[1], y in 1:l[2]]
    for u in indices
        umatrix[u...] = 0.0
        neighbors = __neighbor_units(som, u)
        w = get_unit_weight(som, u)
        for n in neighbors
            v = get_unit_weight(som, n)
            umatrix[u...] += som.dist(collect(v), collect(w))
        end
        umatrix[u...] /= length(neighbors)
    end
    return umatrix
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
    error = 0
    for i=1:size(data, 1)
        input = data[i,:]
        weight = get_unit_weight(som, get_BMU(som, input))
        error += som.dist(vec(weight), vec(input))
    end
    return error / size(data, 1)
end


function reset_state(som::SOM)
    som.t = 0
    som.epoch = 0
    som.rng = MersenneTwister(som.seed)
    init_weights(som)
end


end
