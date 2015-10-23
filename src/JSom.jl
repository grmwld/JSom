module JSom

using DataFrames
using DataStructures
using Distances

import Base.size


export
    SOM,
    _linear_decay,
    _exp_decay,
    _gaussian,
    get_unit_weight,
    set_unit_weight,
    size,
    neighborhood,
    update,
    get_BMU,
    activate,
    sequential_random_epoch,
    sequential_epoch,
    quantize,
    quantization_error,
    bmu_map,
    activation_response



type SOM
    weights::Array{Float64, 3}
    activation_map::Array{Float64, 2}
    η::Float64
    σ::Float64
    λ::Float64
    t::Int
    epoch::Int
    decay::Function
    influence::Function
    dist::Function

    function SOM(x::Int, y::Int, input_len::Int;
                 σ::Float64=1.0, η::Float64=0.5)
        weights = rand(x, y, input_len) * 2 - 1
        this = new()
        this.t = 0
        this.epoch = 0
        this.λ = 0
        this.σ = σ
        this.η = η
        this.weights = broadcast(/, weights, mapslices(norm, weights, [1, 2]))
        this.activation_map = zeros(Float64, (x, y))
        this.decay = _linear_decay
        this.influence = _gaussian
        this.dist = euclidean
        return this
    end

end


function _linear_decay(x::Float64, t::Int, λ::Float64)
    return x / (1 + (t / λ))
end


function _exp_decay(x::Float64, t::Int, λ::Float64)
    return x * exp(-t / λ)
end


function _gaussian(u::Tuple{Int, Int}, bmu::Tuple{Int, Int}, σ::Float64)
    return exp(-(euclidean(collect(u), collect(bmu)))^2 / (2 * σ^2))
end


function size(som::SOM)
    return size(som.activation_map)
end


function get_unit_weight(som::SOM, i::Int, j::Int)
    return som.weights[i, j, :]
end

function get_unit_weight(som::SOM, c::Tuple{Int, Int})
    return get_unit_weight(som, c[1], c[2])
end

function set_unit_weight(som::SOM, i::Int, j::Int, w::Array{Float64})
    som.weights[i, j, :] = w
end

function set_unit_weight(som::SOM, i::Int, j::Int, w::Float64)
    som.weights[i, j, :] = [w]
end

function set_unit_weight(som::SOM, c::Tuple{Int, Int}, w)
    set_unit_weight(som, c[1], c[2], w)
end


function update(som::SOM, input::Array, bmu::Tuple)
    som.t += 1
    input = vec(input)
    η = som.decay(som.η, som.t, som.λ)
    σ = som.decay(som.σ, som.t, som.λ)
    for k in eachindex(som.activation_map)
        u = ind2sub(som.activation_map, k)
        θ = som.influence(u, bmu, σ)
        weight = vec(get_unit_weight(som, u))
        weight = weight + θ * η * (input - weight)
        set_unit_weight(som, u, weight)
    end
end


function get_BMU(som::SOM, input::Array)
    activate(som, input)
    return ind2sub(size(som), indmin(som.activation_map))
end


function activate(som::SOM, input::Array)
    input = reshape(input, (1, 1, size(input, 2)))
    for k in eachindex(som.activation_map)
        i, j = ind2sub(som.activation_map, k)
        weight = get_unit_weight(som, i, j)
        som.activation_map[k] = som.dist(vec(input), vec(weight))
    end
end


function quantize(som::SOM, data::Array)
    q = similar(data)
    for i=1:size(data, 1)
        input = data[i, :]
        bmu = get_BMU(som, input)
        q[i, :] = get_unit_weight(som, bmu)
    end
    return q
end


function activation_response(som::SOM, data::Array)
    a = zeros(size(som))
    for i=1:size(data, 1)
        input = data[i, :]
        bmu = get_BMU(som, input)
        a[bmu...] += 1
    end
    return a
end


function bmu_map(som::SOM, data::Array)
    bmus = DefaultDict(Tuple{Int, Int}, Vector{Vector}, Vector{Vector})
    for i=1:size(data, 1)
        input = data[i, :]
        bmu = get_BMU(som, input)
        push!(bmus[bmu...], vec(input))
    end
    return bmus
end


function sequential_random_epoch(som::SOM, data::Array, num_iter::Int)
    init_λ(som, num_iter)
    for t = 0:num_iter
        i = rand(1:size(data, 1))
        input = data[i, :]
        update(som, input, get_BMU(som, input))
    end
end


function sequential_epoch(som::SOM, data::Array)
    num_iter = size(data, 1)
    init_λ(som, num_iter)
    for t in shuffle(collect(1:num_iter))
        input = data[t, :]
        update(som, input, get_BMU(som, input))
    end
end


function quantization_error(som::SOM, data::Array)
    error = 0
    for i=1:size(data, 1)
        input = data[i, :]
        weight = get_unit_weight(som, get_BMU(som, input))
        error += som.dist(vec(weight), vec(input))
    end
    return error / size(data, 1)
end


function init_λ(som::SOM, num_iter::Int)
    som.λ = (som.t + num_iter) / 2
end


end
