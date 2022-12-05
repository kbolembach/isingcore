using Random
using DelimitedFiles
using ArgParse
using Dates
using Plots
plotlyjs()


"""Perform the Metropolis algorithm on lattice spins. 
Calculate the difference in energy ΔU if we were to flip the spin at each site 
(x, y) in the lattice. In a two dimensional setting ΔU can be equal to: -4, -2, 
0, 2, 4. For ΔU less or equal to zero, we flip the spin; for positive ΔU we flip 
the spin according to cached probability of exp(-2ΔU / T). Performing this 
algorithm on every site in one step is called a sweep."""
function sweep!(lattice::Matrix{Int64}, 
    L_x::Int64, 
    L_y::Int64, 
    next::Vector{Int64},
    prev::Vector{Int64},
    probability::Vector{Float64})

    # Normal sweep
    # for x in range(1, stop=L_x)
    #     for y in range(1, stop=L_y)
    #         @inbounds ΔU = lattice[x, y] * (lattice[next[x], y] + lattice[x, next[y]] + lattice[prev[x], y] + lattice[x, prev[y]]) 
    #         if ΔU <= 0 
    #             @inbounds lattice[x, y] *= -1
    #         elseif rand() < probability[ΔU]
    #             @inbounds lattice[x, y] *= -1
    #         end
    #     end
    # end

    # Sweep but in random order
    indices = Iterators.product(randperm(L_x), randperm(L_y)) |> collect
    for (x, y) in indices
        @inbounds neighbours = lattice[next[x], y] + lattice[x, next[y]] + lattice[prev[x], y] + lattice[x, prev[y]]
        @inbounds ΔU = lattice[x, y] * neighbours
        if ΔU <= 0 
            @inbounds lattice[x, y] *= -1
        elseif rand() < probability[ΔU]
            @inbounds lattice[x, y] *= -1
        end
    end
end


"""Calculate lattice energy based on spins."""
function get_energy(lattice::Matrix{Int64}, 
    L_x::Int64, 
    L_y::Int64, 
    next::Vector{Int64})

    out = 0
    for x in range(1, stop=L_x)
        for y in range(1, stop=L_y)
            @inbounds out += -1 * lattice[x, y] * (lattice[next[x], y] + lattice[x, next[y]])
        end
    end
    return out
end

"""Simulate a simple Ising Model in two dimensions using the Metropolis 
algorithm. Input the lattice size as a tuple of two integers and a 
temperature. MCS is Monte Carlo Steps - number of sweeps the algorithm 
will perform. 

The algorithm will start gathering info on lattice parameters after 
reaching THRESHOLD number of steps and will update data once every
STEP steps."""
function ising(lattice_size::Tuple{Int64, Int64}, 
    temperature::Float64,
    MCS::Int64, 
    THRESHOLD::Int64,
    STEP_SIZE::Int64)

    
    lattice = ones(Int64, lattice_size)
    # For a random lattice:
    # lattice = rand!(lattice, [-1, 1])

    ising(lattice, temperature, MCS, THRESHOLD, STEP_SIZE)
end

function ising(lattice::Matrix{Int64}, 
    temperature::Float64,
    MCS::Int64, 
    THRESHOLD::Int64,
    STEP_SIZE::Int64)

    lattice_size = size(lattice)
    L_x, L_y = lattice_size
    V = reduce(*, lattice_size)
    
    # we are gonna assume for now, that L_x == L_y
    next = [collect(2:L_x); [1]]
    prev = [[L_x]; collect(1:L_x-1)]

    probabilities = exp.([0.0, -4.0, 0.0, -8.0] ./ temperature)
    energies_arr = []
    magnetizations_arr = []

    # Equilibrate for THRESHOLD steps:
    for _ in range(1, stop=THRESHOLD)
        sweep!(lattice, L_x, L_y, next, prev, probabilities)
    end

    for i_MCS in range(THRESHOLD, stop=MCS)
        # In each Monte Carlo step, sweep through the entire lattice
        # and apply the Metropolis algorithm to flip spins accordingly.
        sweep!(lattice, L_x, L_y, next, prev, probabilities)

        # Record parameters every STEP number of steps.
        if mod(i_MCS, STEP_SIZE) == 0
            current_energy = get_energy(lattice, L_x, L_y, next)
            push!(energies_arr, current_energy)
        
            current_magnetization = reduce(+, lattice) / V
            push!(magnetizations_arr, current_magnetization)
        end
    end

    return lattice, magnetizations_arr, energies_arr
end

"""Calculate variance of data."""
function variance(data)
    div = length(data)
    E = reduce(+, data)
    E2 = reduce(+, [x^2 for x in data])

    return ((E2/div) - (E/div)^2)
end

"""Get specific heat from an array of recorded energies for a given temperature T."""
function get_specific_heat(energies, temperature)
    return variance(energies) / temperature
end


"""Get magnetic susceptibility from an array of recorded magnetizations for a given temperature T."""
function get_susceptibility(magnetizations, temperature)
    return variance(magnetizations) / temperature
end


"""Get absolute magnetic susceptibility from an array of recorded magnetizations for a given temperature T."""
function get_abs_susceptibility(magnetizations, temperature)
    return variance(abs.(magnetizations)) / temperature
end


"""Helper function to parse arguments from commandline."""
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--path"
            help = "continue simulation upon some data (in a new folder); specify path to data"
            default = ""
        "--size"
            help = "size of the 2-dimensional lattice"
            default = 14
            arg_type = Int
        "--threshold"
            help = "amount of Monte Carlo Steps to equilibrate the lattice"
            default = 100000
            arg_type = Int
        "--mcs"
            help = "number of total Monte Carlo Steps in simulation"
            default = 300000
            arg_type = Int
        "--step-size"
            help = "how often to collect data after equilibration"
            default = 100
            arg_type = Int
        "--temp-min"
            help = "lowest temperature in range"
            default = 1.0
            arg_type = Float64
        "--temp-max"
            help = "maximum temperature in range"
            default = 3.0
            arg_type = Float64
        "--temp-step"
            help = "temperature step"
            default = 0.1
            arg_type = Float64
    end
    return parse_args(s)
end


"""Helper function to draw graphs of data and save them with a given name."""
function save_graph_and_data(path, x, data, name)
    pl = plot(x, data)
    savefig(pl, path * "$name plot")
    writedlm(path * "$name.csv", data)
end


function main()
    t_start_program = now()
    t_start_str = Dates.format(t_start_program, "dd-mm_HH-MM")
    n = Threads.nthreads()
    # Careful: nw is volatile, used to display time and date for every 
    # useful println in the program and changes at every point.
    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Using $n threads.")
    parsed_args = parse_commandline()

    OLD_PATH                = parsed_args["path"]
    CONTINUE_FLAG           = OLD_PATH == "" ? false : true
    OLD_MAGNETIZATIONS_FLAG = isdir(OLD_PATH * "/magnetizations/")
    OLD_ENERGIES_FLAG       = isdir(OLD_PATH * "/energies/")
    LATTICE_SIZE            = (parsed_args["size"], parsed_args["size"])
    MCS                     = parsed_args["mcs"]
    THRESHOLD               = parsed_args["threshold"]
    STEP_SIZE               = parsed_args["step-size"]
    TEMP_MIN                = parsed_args["temp-min"]
    TEMP_MAX                = parsed_args["temp-max"]
    TEMP_STEP               = parsed_args["temp-step"]

    if CONTINUE_FLAG 
        if !isdir(OLD_PATH)
            println("ERROR: Specified path doesn't exist.")
            return 0
        end
        i_start = findfirst("MCS=", OLD_PATH)[end] + 1
        i_end = findfirst(" STEP_SIZE", OLD_PATH)[1] - 1
        total_MCS = parse(Int64, OLD_PATH[i_start:i_end]) + MCS
    else
        total_MCS = MCS
    end

    temp_range = collect(range(TEMP_MIN, stop=TEMP_MAX, step=TEMP_STEP))
    temp_length = length(temp_range)

    path = "./isingcore/" * t_start_str * " L=$LATTICE_SIZE MCS=$total_MCS T=$TEMP_MIN _$TEMP_STEP _$TEMP_MAX/"
    mkpath(path * "/lattices")
    mkpath(path * "/magnetizations")
    mkpath(path * "/energies")
    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Directories created.")
    
    ##########################################################################################
    # Load/initialize data as needed
    
    energies_array              = Array{Float64}(undef, temp_length)
    abs_magnetizations_array    = Array{Float64}(undef, temp_length)
    χ_array                     = Array{Float64}(undef, temp_length)
    abs_χ_array                 = Array{Float64}(undef, temp_length)
    specific_heat_array         = Array{Float64}(undef, temp_length)
    exec_times                  = Array{String}(undef, temp_length)
    t1_run                      = Array{Millisecond}(undef, temp_length)
    t2_run                      = Array{Millisecond}(undef, temp_length)

    lattices_atlas          = Array{Matrix{Int64}}(undef, temp_length)
    magnetizations_atlas    = Array{Array{Float64}}(undef, temp_length)
    energies_atlas          = Array{Array{Float64}}(undef, temp_length)

    if CONTINUE_FLAG
        old_lattices_atlas          = Array{Matrix{Int64}}(undef, temp_length)
        old_magnetizations_atlas    = Array{Array{Float64}}(undef, temp_length)
        old_energies_atlas          = Array{Array{Float64}}(undef, temp_length)
        
        t1_load = now()
        for idx in eachindex(temp_range)
            T = temp_range[idx]
            old_lattices_atlas[idx] = readdlm(OLD_PATH * "/lattices/T=$T.csv", ',', Int64)

            if OLD_MAGNETIZATIONS_FLAG
                old_magnetizations_atlas[idx] = readdlm(OLD_PATH * "/magnetizations/T=$T.csv")
            end
            if OLD_ENERGIES_FLAG
                old_energies_atlas[idx] = readdlm(OLD_PATH * "/energies/T=$T.csv")
            end
        end

        t2_load = now()
        t_total_load = string(t2_load - t1_load)
        nw = Dates.format(now(), "mm-dd HH:MM:SS")
        println("$nw  Loaded data in time: $t_total_load")
    end

    ##########################################################################################
    # Run the algorithm in a parallelized loop
    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Running the algorithm, L=$LATTICE_SIZE, T=$TEMP_MIN _$TEMP_STEP _$TEMP_MAX.")
    # counter = Threads.Atomic{Int}(0);
    # count_against = collect(0:div(length(temp_range), 10):length(temp_range))
    
    Threads.@threads for idx in eachindex(temp_range)
        current_temp = temp_range[idx]
        t1_run[idx] = now()
        if CONTINUE_FLAG
            lattice, magn_arr_new, ener_arr_new = ising(old_lattices_atlas[idx], current_temp, MCS, THRESHOLD, STEP_SIZE)

            if OLD_MAGNETIZATIONS_FLAG
                magn_arr = [old_magnetizations_atlas[idx]; magn_arr_new]
            else
                magn_arr = magn_arr_new
            end

            if OLD_ENERGIES_FLAG
                ener_arr = [old_energies_atlas[idx]; ener_arr_new]
            else
                ener_arr = ener_arr_new
            end

        else
            lattice, magn_arr, ener_arr = ising(LATTICE_SIZE, current_temp, MCS, THRESHOLD, STEP_SIZE)
        end
        # Threads.atomic_add!(counter, 1)

        susceptibility      = get_susceptibility(magn_arr, current_temp)
        abs_susceptibility  = get_abs_susceptibility(magn_arr, current_temp)
        specific_heat       = get_specific_heat(ener_arr, current_temp)

        energies_array[idx]             = reduce(+, ener_arr)/length(ener_arr)
        abs_magnetizations_array[idx]   = reduce(+, abs.(magn_arr))/length(magn_arr)
        specific_heat_array[idx]        = specific_heat
        abs_χ_array[idx]                = abs_susceptibility
        χ_array[idx]                    = susceptibility
        
        lattices_atlas[idx]         = lattice
        magnetizations_atlas[idx]   = magn_arr
        energies_atlas[idx]         = ener_arr
        
        t2_run[idx] = now()
        exec_times[idx] = string(t2_run[idx] - t1_run[idx])
        
        # if counter[] in count_against
        #     precentage = div((100counter[]), temp_length)
        #     nw = Dates.format(now(), "mm-dd HH:MM:SS")
        #     println("$nw  Done $precentage%,      L=$LATTICE_SIZE, T=$TEMP_MIN _$TEMP_STEP _$TEMP_MAX. ")
        # end
        println("done T=" * string(current_temp) * " ; time = " * exec_times[idx])
    end
    
    ##########################################################################################
    # Save data to files

    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Saving plots and data...")
    for idx in eachindex(temp_range)
        T = temp_range[idx]
        writedlm(path * "/lattices/T=$T.csv", lattices_atlas[idx], ',')
        writedlm(path * "/magnetizations/T=$T.csv", magnetizations_atlas[idx], ',')
        writedlm(path * "/energies/T=$T.csv", energies_atlas[idx], ',')
    end

    save_graph_and_data(path, temp_range, abs_magnetizations_array, "abs_magnetization")
    save_graph_and_data(path, temp_range, energies_array, "energy")
    save_graph_and_data(path, temp_range, χ_array, "susceptibility")
    save_graph_and_data(path, temp_range, abs_χ_array, "abs_susceptibility")
    save_graph_and_data(path, temp_range, specific_heat_array, "specific_heat")

    writedlm(path * "exec_time.txt", exec_times, '\n')
    
    # Write the report and exit
    t_end_program = now()
    t_end_str = string(t_end_program)
    total_exec_time = string(t_end_program - t_start_program)
    DIV = div(MCS-THRESHOLD, STEP_SIZE) + 1

    report_str = """Report 
    program start time = $t_start_str
    program end time = $t_end_str
    number of threads = $n

    lattice size = $LATTICE_SIZE
    MCS = $MCS
    total_MCS = $total_MCS
    threshold = $THRESHOLD
    step size = $STEP_SIZE
    temp min = $TEMP_MIN
    temp max = $TEMP_MAX
    temp step = $TEMP_STEP

    DIV = $DIV 
    total execution time = $total_exec_time
    """

    open(path * "report.txt", "w") do file
        write(file, report_str)
    end
    nw = Dates.format(now(), "mm-dd HH:MM:SS")
    println("$nw  Finished.")
    return 0
end

main()