importall GeneticAlgorithms

using GeneticAlgorithms

require("Jsom/src/ParamOptimization.jl")


function GeneticAlgorithms.crossover_population(model::GAmodel, groupings)
    model.gen_num += 1
    for group in groupings
        parents = { model.population[i] for i in group }
        entity = model.ga.crossover(parents)
        push!(model.population, entity)
        push!(model.pop_data, GeneticAlgorithms.EntityData(entity, model.gen_num))
    end
    GeneticAlgorithms.evaluate_population(model)
    model.population = model.population[1:length(model.population)/2]
end


model = runga(param_optimisation, initial_pop_size = 100)

@show population(model)[1].fitness
@show population(model)[1].som.σ
@show population(model)[1].som.η
@show population(model)[1].som.decay
@show population(model)[1].som.influence
@show population(model)[1].som.dist
