using JSom
using Base.Test

srand(1)


@testset "Grid SOM" begin
    gsom = GridSOM(5, 5, 1)
    gsom.weights = zeros(5, 5, 1)
    set_unit_weight(gsom, 3, 4, 5.0)
    set_unit_weight(gsom, 2, 2, 2.0)

    @testset "neighboring units" begin
        n1 = sort(JSom.__neighbor_units(gsom, (2,2)))
        n2 = sort(JSom.__neighbor_units(gsom, (1,5)))
        x, y = (2,2)
        @test n1 == sort([
            (x-1, y-1), (x-1, y), (x-1, y+1),
            (x, y-1),             (x, y+1),
            (x+1, y-1), (x+1, y), (x+1, y+1)
        ])
        x, y = (1,5)
        @test n2 == sort([
            (x, y-1),
            (x+1, y-1), (x+1, y)
        ])
    end

    @testset "U-matrix" begin
        u = umatrix(gsom)
        @test u[3,4] == get_unit_weight(gsom, (3,4))[1]
        @test u[2,2] == get_unit_weight(gsom, (2,2))[1]
        @test u[2,3] == mean([2, 5, 0, 0, 0, 0, 0, 0])
    end

    @testset "neighborhood functions" begin
        L = 5
        indices = [(x,y) for x in 1:L, y in 1:L]
        @testset "gaussian" begin
            gsom.ħ = Gaussian_Neighborhood()
            bell = reshape([ħ(gsom, u, (2,2), 1.0) for u in indices], (L,L))
            @test maximum(bell) == 1.0
            @test indmax(bell) == 7
            @test bell[2,1] == bell[2,3] == bell[1,2] == bell[3,2]
            @test bell[1,1] == bell[1,3] == bell[3,1] == bell[3,3]
            @test bell[2,2] ≥ bell[2,3] ≥ bell[2,4] ≥ bell[2,5]
        end

        @testset "mexican hat" begin
            gsom.ħ = Ricker_Neighborhood()
            bell = reshape([ħ(gsom, u, (2,2), 1.0) for u in indices], (L,L))
            @test maximum(bell) == 1.0
            @test indmax(bell) == 7
            @test bell[2,1] == bell[2,3] == bell[1,2] == bell[3,2] == 0
            @test bell[1,1] == bell[1,3] == bell[3,1] == bell[3,3] < 0
            @test bell[3,3] ≤ bell[4,4] ≤ bell[5,5]
        end

        @testset "triangular" begin
            gsom.ħ = Triangular_Neighborhood()
            bell = reshape([ħ(gsom, u, (2,2), 2.0) for u in indices], (L,L))
            @test maximum(bell) == 1.0
            @test indmax(bell) == 7
            @test bell[2,1] == bell[2,3] == bell[1,2] == bell[3,2] == 0.5
            @test bell[3,4] == bell[3,5] == 0.0
        end
    end
    
    @testset "bmu_map" begin
        bmus = bmu_map(gsom, [5,2])
        @test bmus[(3,4)] == Vector[[5]]
        @test bmus[(2,2)] == Vector[[2]]
        @test bmus[(1,1)] == []
        @test bmus[(2,4)] == []
    end

    @testset "activation_response" begin
        response = activation_response(gsom, [5,2])
        @test response[3,4] == 1
        @test response[2,2] == 1
        @test response[1,1] == 0
        @test response[2,4] == 0
    end

    @testset "activate" begin
        activate(gsom, [5.0,])
        @test indmin(gsom.activation_map) == 18
    end

    @testset "quantize" begin
        q = quantize(gsom, [4,2])
        @test q[1] == 5.0
        @test q[2] == 2.0
    end

    @testset "quantization_error" begin
        @test quantization_error(gsom, [5,2]) == 0.0
        @test quantization_error(gsom, [4,2]) == 0.5
    end
end


@testset "Hex SOM" begin
    hsom1 = HexSOM(5, 5, 1)
    hsom1.weights = zeros(5, 5, 1)
    set_unit_weight(hsom1, 3, 4, 5.0)
    set_unit_weight(hsom1, 2, 2, 2.0)

    @testset "neighboring units" begin
        n1 = sort(JSom.__neighbor_units(hsom1, (2,2)))
        n2 = sort(JSom.__neighbor_units(hsom1, (1,5)))
        x, y = (2,2)
        @test n1 == sort([
                        (x-1, y), (x-1, y+1),
            (x, y-1),             (x, y+1),
            (x+1, y-1), (x+1, y)
        ])
        x, y = (1,5)
        @test n2 == sort([
            (x, y-1),
            (x+1, y-1), (x+1, y)])
    end
end

@testset "Generic SOM" begin
    data = rand(4, 2)
    @testset "random seed" begin
        som1 = GridSOM(5, 5, 2, seed=1)
        som2 = GridSOM(5, 5, 2, seed=1)
        @test som1.weights == som2.weights
        sequential_random_epoch(som1, data, 5)
        sequential_random_epoch(som2, data, 5)
        @test som1.weights == som2.weights
    end

    @testset "reset" begin
        som = GridSOM(5, 5, 2, seed=1)
        sequential_random_epoch(som, data, 5)
        q1 = quantization_error(som, data)
        reset_state(som)
        sequential_random_epoch(som, data, 5)
        q2 = quantization_error(som, data)
        @test q1 == q2
    end

    @testset "training" begin
        @testset "sequential_epoch" begin
            som = GridSOM(5, 5, 2)
            data = rand(4, 2)
            q1 = quantization_error(som, data)
            sequential_epoch(som, data)
            q2 = quantization_error(som, data)
            sequential_epoch(som, data)
            q3 = quantization_error(som, data)
            @test q1 ≥ q2
            @test q2 ≥ q3
        end

        @testset "sequential_random_epoch" begin
            som = GridSOM(5, 5, 2)
            data = rand(4, 2)
            q1 = quantization_error(som, data)
            sequential_random_epoch(som, data, 5)
            q2 = quantization_error(som, data)
            sequential_random_epoch(som, data, 5)
            q3 = quantization_error(som, data)
            @test q1 ≥ q2
            @test q2 ≥ q3
        end
    end
end


@testset "decay functions" begin
    @testset "inverse" begin
        @test _τ_inverse(1., 2, 3.) == 1 / (1 + 2/3)
    end

    @testset "exponential" begin
        @test _τ_exponential(1., 2, 3.) == 1 * exp(-2/3)
    end
end

