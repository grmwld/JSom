using JSom
using Base.Test

srand(1)

@testset "JSom tests" begin
    som1 = SOM(5, 5, 1)
    som1.weights = zeros(5, 5, 1)
    som1.weights[3, 4, 1] = 5.0
    som1.weights[2, 2, 1] = 2.0

    @testset "decay_function" begin
        @test som1.decay_function(1, 2, 3) == 1/(1+2/3)
    end

    @testset "activate" begin
        activate(som1, [5.0,])
        @test indmin(som1.activation_map) == 18
    end

    @testset "quantization_error" begin
        @test quantization_error(som1, [6, 3]) == 0.0
        @test quantization_error(som1, [5, 2]) == 0.5
    end

    @testset "train_random" begin
        som2 = SOM(5, 5, 2)
        data = [4 2 ; 3 1]
        train_random(som2, data, 10)
    end
end
