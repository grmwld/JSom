module JSom

    #=using ArgParse=#
    #=using DataFrames=#
    using Base.Test
    using Distances

    import Base.size


    type SOM
        learning_rate::Float64
        sigma::Float64
        weights::Array{Float64, 3}
        activation_map::Array{Float64, 2}
        neigx::Array{Int, 1}
        neigy::Array{Int, 1}
        T::Float64
        decay_function::Function

        function SOM(x::Int, y::Int, input_len::Int;
                     sigma::Float64=1.0, learning_rate::Float64=0.5)
            weights = rand(x, y, input_len) * 2 - 1
            this = new()
            this.T = 0
            this.sigma = sigma
            this.learning_rate = learning_rate
            this.weights = broadcast(/, weights, mapslices(norm, weights, [1, 2]))
            this.activation_map = zeros(Float64, (x, y))
            this.neigx = range(1, x)
            this.neigy = range(1, y)
            this.decay_function = (x, t, m) -> x / (1 + (t / m))
            return this
        end

        function SOM(x::Int, y::Int, input_len::Int, decay_function::Function;
                     sigma::Float64=1.0, learning_rate::Float64=0.5)
            this = SOM(x, y, input_len, sigma=sigma, learning_rate=learning_rate)
            this.decay_function = decay_function
            return this
        end
    end


    function size(som::SOM)
        return size(som.activation_map)
    end


    function neighborhood(som, c, sigma)
        d = 2 * pi * sigma^2
        ax = exp(-(som.neigx - c[1]).^2 / d)
        ay = exp(-(som.neigy - c[2]).^2 / d)
        return ax * ay'
    end


    function update(som::SOM, sample::Array, winner::Tuple, epoch::Int)
        eta = som.decay_function(som.learning_rate, epoch, som.T)
        sigma = som.decay_function(som.sigma, epoch, som.T)
        g = neighborhood(som, winner, sigma) * eta
        sample = reshape(sample, (1, 1, length(sample)))
        for k in eachindex(g)
            i, j = ind2sub(g, k)
            som.weights[i, j, :] += g[i, j] * (sample - som.weights[i, j, :])
            som.weights[i, j, :] /= norm(vec(som.weights[i, j, :]))
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


    function train_random(som::SOM, data::Array, num_epoch::Int)
        init_T(som, num_epoch)
        for epoch = 0:num_epoch
            i = rand(1:size(data, 1))
            sample = data[i, :]
            update(som, sample, winner(som, sample), epoch)
        end
    end


    function quantization_error(som::SOM, data::Array)
        error = 0
        for i=1:size(data, 1)
            sample = data[i, :]
            w = winner(som, sample)
            error += euclidian(vec(som.weights[w[1], w[2] :]), sample)
        end
        return error / length(data)
    end


    function init_T(som::SOM, num_epoch::Int)
        som.T = num_epoch / 2
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


    end

end
