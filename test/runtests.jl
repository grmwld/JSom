using JSom
using Base.Test

srand(1)

@testset "JSom tests" begin
    som1 = SOM(5, 5, 1)
    som1.weights = zeros(5, 5, 1)
    set_unit_weight(som1, 3, 4, 5.0)
    set_unit_weight(som1, 2, 2, 2.0)

    @testset "decay_function" begin
        @test _linear_decay(1., 2, 3.) == 1/(1+2/3)
        @test _exp_decay(1., 2, 3.) == 1 * exp(-2 / 3)
    end
    
    @testset "bmu_map" begin
        bmus = bmu_map(som1, [5, 2])
        @test bmus[(3, 4)] == Vector[[5]]
        @test bmus[(2, 2)] == Vector[[2]]
        @test bmus[(1, 1)] == []
        @test bmus[(2, 4)] == []
    end

    @testset "activation_response" begin
        response = activation_response(som1, [5, 2])
        @test response[3, 4] == 1
        @test response[2, 2] == 1
        @test response[1, 1] == 0
        @test response[2, 4] == 0
    end

    @testset "activate" begin
        activate(som1, [5.0,])
        @test indmin(som1.activation_map) == 18
    end

    @testset "quantize" begin
        q = quantize(som1, [4, 2])
        @test q[1] == 5.0
        @test q[2] == 2.0
    end

    @testset "quantization_error" begin
        @test quantization_error(som1, [5, 2]) == 0.0
        @test quantization_error(som1, [4, 2]) == 0.5
    end

    @testset "training" begin
        @testset "sequential_epoch" begin
            som2 = SOM(5, 5, 2)
            data = rand(4, 2)
            q1 = quantization_error(som2, data)
            sequential_epoch(som2, data)
            q2 = quantization_error(som2, data)
            sequential_epoch(som2, data)
            q3 = quantization_error(som2, data)
            @test q1 ≥ q2
            @test q2 ≥ q3
        end

        @testset "sequential_random_epoch" begin
            som2 = SOM(5, 5, 2)
            data = rand(4, 2)
            q1 = quantization_error(som2, data)
            sequential_random_epoch(som2, data, 5)
            q2 = quantization_error(som2, data)
            sequential_random_epoch(som2, data, 5)
            q3 = quantization_error(som2, data)
            @test q1 ≥ q2
            @test q2 ≥ q3
        end
    end
end
