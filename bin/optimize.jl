using GeneticAlgorithms
using JSom
require("Jsom/src/ParamOptimization.jl")

model = runga(param_optimisation, initial_pop_size = 100)

@show population(model)[1].fitness
@show population(model)[1].som.σ
@show population(model)[1].som.η
@show population(model)[1].som.decay
@show population(model)[1].som.influence
@show population(model)[1].som.dist
