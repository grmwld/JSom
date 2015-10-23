module param_optimisation

import Base.isless

using GeneticAlgorithms
using StatsBase
using Distances
using JSom

srand(1234)
DATA1 = [rand(100, 8); rand(100, 8)*3]
DATA2 = [rand(100, 8); rand(100, 8)*3]
srand()

influence_fns = [_gaussian]
decay_fns = [_inverse_decay, _exp_decay]
dist_fns = [euclidean, cosine_dist, corr_dist]


type SOM_Entity <: Entity
    som_params::Array
    som::SOM
    fitness

    function SOM_Entity(som_params::Array, dimensions::Tuple)
        this = new()
        this.som = SOM(dimensions...)
        update_entity(this, som_params)
        return this
    end
end


function create_entity(num)
    σ = rand() * 10
    η = rand()
    decay = sample(decay_fns)
    influence = sample(influence_fns)
    dist = sample(dist_fns)
    return SOM_Entity([σ, η, decay, influence, dist], (10, 10, 8))
end


function update_entity(ent, params)
    ent.som_params = params
    σ, η, decay, influence, dist = ent.som_params
    ent.som.σ = σ
    ent.som.η = η
    ent.som.decay = decay
    ent.som.influence = influence
    ent.som.dist = dist
    JSom.reset(ent.som)
    sequential_epoch(ent.som, DATA1, 3, 1234)
end


function fitness(ent)
    score = quantization_error(ent.som, DATA1)
    return score
end


function isless(lhs::SOM_Entity, rhs::SOM_Entity)
    lhs.fitness > rhs.fitness
end


function group_entities(pop)
    @show pop[1].fitness
    @show pop[1].som.σ
    @show pop[1].som.η
    @show pop[1].som.decay
    @show pop[1].som.influence
    @show pop[1].som.dist
    println("---")
    if pop[1].fitness <= 0.01
        return
    end
    c = [shuffle(collect(1:length(pop))) shuffle(collect(1:length(pop)))]
    for i in 1:size(c, 1)
        produce(c[i, :])
    end
end


function crossover(group)
    som_params = similar(group[1].som_params)
    num_parents = length(group)
    for i in 1:length(group[1].som_params)
        parent = (rand(UInt) % num_parents) + 1
        som_params[i] = group[parent].som_params[i]
    end
    child = SOM_Entity(som_params, (10, 10, 8))
    return child
end


function mutate(ent)
    rand(Float64) < 0.90 && return
    i = shuffle(collect(1:length(ent.som_params)))[1]
    som_params = ent.som_params
    if i == 1
        som_params[i] = rand() * 10
    elseif i == 2
        som_params[i] = rand()
    elseif i == 3
        som_params[i] = sample(decay_fns)
    elseif i == 4
        som_params[i] = sample(influence_fns)
    elseif i == 5
        som_params[i] = sample(dist_fns)
    end
    update_entity(ent, som_params)
end

end
