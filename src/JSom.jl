module JSom

#=using ArgParse=#
#=using DataFrames=#
using Distances

import Base.size


export SOM
export get_unit_weight
export set_unit_weight
export size
export neighborhood
export update
export winner
export activate
export train_random
export quantization_error


type SOM
    learning_rate::Float64
    sigma::Float64
    weights::Array{Float64, 3}
    activation_map::Array{Float64, 2}
    neigx::Array{Int, 1}
    neigy::Array{Int, 1}
    T::Float64
    t::Int
    epochs::Int
    decay_function::Function
    dist_function::Function

    function SOM(x::Int, y::Int, input_len::Int;
                 sigma::Float64=1.0, learning_rate::Float64=0.5)
        weights = rand(x, y, input_len) * 2 - 1
        this = new()
        this.T = 0
        this.t = 0
        this.epochs = 0
        this.sigma = sigma
        this.learning_rate = learning_rate
        this.weights = broadcast(/, weights, mapslices(norm, weights, [1, 2]))
        this.activation_map = zeros(Float64, (x, y))
        this.neigx = range(1, x)
        this.neigy = range(1, y)
        this.decay_function = (x, t, m) -> x / (1 + (t / m))
        this.dist_function = (a, b) -> euclidean(a, b)
        return this
    end

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


function neighborhood(som, c, σ)
    d = 2 * pi * σ^2
    ax = exp(-(som.neigx - c[1]).^2 / d)
    ay = exp(-(som.neigy - c[2]).^2 / d)
    return ax * ay'
end


function update(som::SOM, sample::Array, winner::Tuple)
    som.t += 1
    η = som.decay_function(som.learning_rate, som.t, som.T)
    σ = som.decay_function(som.sigma, som.t, som.T)
    g = neighborhood(som, winner, σ) * η
    sample = reshape(sample, (1, 1, length(sample)))
    for k in eachindex(g)
        i, j = ind2sub(g, k)
        weight = get_unit_weight(som, i, j)
        weight += g[i, j] * (sample - weight)
        weight /= norm(vec(weight))
        set_unit_weight(som, i, j, weight)
        #=som.weights[i, j, :] += g[i, j] * (sample - weight)=#
        #=som.weights[i, j, :] /= =#
    end
end


function winner(som::SOM, sample::Array)
    activate(som, sample)
    return ind2sub(size(som), indmin(som.activation_map))
end


function activate(som::SOM, sample::Array)
    sample = reshape(sample, (1, 1, size(sample, 2)))
    s = broadcast(-, sample, som.weights)
    for k in eachindex(som.activation_map)
        i, j = ind2sub(som.activation_map, k)
        som.activation_map[k] = norm(vec(s[i, j, :]))
    end
end


function train_random(som::SOM, data::Array, num_iter::Int)
    init_T(som, num_iter)
    for t = 0:num_iter
        i = rand(1:size(data, 1))
        sample = data[i, :]
        update(som, sample, winner(som, sample))
    end
end


function quantization_error(som::SOM, data::Array)
    error = 0
    for i=1:size(data, 1)
        sample = data[i, :]
        weight = get_unit_weight(som, winner(som, sample))
        error += som.dist_function(vec(weight), vec(sample))
    end
    return error / size(data, 1)
end


function init_T(som::SOM, num_iter::Int)
    som.T = num_iter / 2
end



#=function parse_commandline()=#
    #=s = ArgParseSettings()=#
    #=@add_arg_table s begin=#
        #="--infile", "-i"=#
            #=help = "Input file"=#
        #="--outfile", "-o"=#
            #=help = "Output file"=#
    #=end=#
    #=return parse_args(s)=#
#=end=#

function main()
    srand(1) 
    #=data = rand(1000, 5)=#

    #=som = SOM(4, 4, 5, sigma=1.0, learning_rate=0.5)=#

    #=@time train_random(som, data, 100)=#

    #=println(som.activation_map)=#
end

end
