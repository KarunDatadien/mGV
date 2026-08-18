module mGV

using NetCDF
using NCDatasets
using Zarr
using ProgressMeter
using Dates
using LinearAlgebra  # Need this?
using TimerOutputs
using MacroTools
using Printf
using Statistics
using KernelAbstractions
using Adapt: adapt, @adapt_structure

include("config.jl")
using .Config: load_config, Cfg
include("backend_setup.jl")
include("constants.jl")
using .Constants: PhysConsts, SimConsts, SnowConsts

include("reader.jl")
include("parameters.jl")
include("physics.jl")
include("snow.jl")
include("soil.jl")
include("routing.jl")
include("temperature.jl")
include("postprocess.jl")

"""Validate the path of a file relative to the given directory."""
function validate_path(file, dir)
    file = abspath(joinpath(dir, file))
    if endswith(file, "_")
        files = readdir(dirname(file))
        n_matching_files = sum(startswith.(files, basename(file)))
        if n_matching_files < 1
            error("No files found in ", dirname(file), "starting with", basename(file))
        end
    elseif !isfile(file)
        error("Cannot find file '$file'")
    end
    return file
end

"""
Clock struct for timekeeping.

Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
mutable struct Clock{T}
    time::T
    iteration::Int
    dt::Second
end

"""
Advance clock one step in time.

Note: at iteration 0 the clock is not advanced in time,
to have the model start time at iteration 1.
"""
function advance!(clock)
    if clock.iteration != 0
        clock.time += clock.dt
    end
    clock.iteration += 1
    return nothing
end

"""Initialize clock based on config"""
function Clock(config::Cfg)
    return Clock(
        DateTime(config.start_year),
        0,
        Second(config.timestep)
    )
end


"""
mGV model state.

The model state includes all information the model's update functions
require.
"""
@kwdef struct Model
    config::Cfg    # all configuration options
    clock::Clock   # to keep track of simulation time
    grid_parameters::GridParameters
    vegetation_parameters::VegetationParameters
    soil_parameters::SoilParameters
    surface_energy_variables::SurfaceEnergyVariables
    canopy_variables::CanopyVariables
    soil_variables::SoilVariables
    snow_variables::SnowVariables
    forcing_variables::ForcingVariables
    forcing_readers::ForcingReaders
    routing::RoutingState
    # writer::W                       # writes model output
end

include("energy_balance.jl")
include("evaporation.jl")

function Model(config::Cfg)
    clock = Clock(config)
    forcing_readers, forcing_variables = initialize_forcing(config)
    grid_parameters, vegetation_parameters, soil_parameters = read_parameters(config)

    # Check dim order!
    nx = length(grid_parameters.longitude)
    ny = length(grid_parameters.latitude)
    nveg = cfg.nveg  # vegetation types
    nbands = cfg.nbands  # snow bands
    nlayers = size(soil_parameters.depth, 3) # derive soil layers from input data
    grid_dims = (nx, ny)
    tile_dims = (nx, ny, nbands, nveg)
    soil_dims = (nx, ny, nlayers)

    surface_energy_variables = SurfaceEnergyVariables(grid_dims, tile_dims)
    canopy_variables = CanopyVariables(tile_dims)
    soil_variables = SoilVariables(grid_dims, soil_dims)
    snow_variables = SnowVariables(nx, ny, nbands, nveg)
    routing = RoutingState(config, grid_parameters.elevation)

    # Move data to backend during model initialization
    if backend_name != "CPU"
        grid_parameters = adapt(ArrayType, grid_parameters)
        vegetation_parameters = adapt(ArrayType, vegetation_parameters)
        soil_parameters = adapt(ArrayType, soil_parameters)
        surface_energy_variables = adapt(ArrayType, surface_energy_variables)
        canopy_variables = adapt(ArrayType, canopy_variables)
        soil_variables = adapt(ArrayType, soil_variables)
        snow_variables = adapt(ArrayType, snow_variables)
        forcing_variables = adapt(ArrayType, forcing_variables)
        routing = adapt(ArrayType, routing)
    end

    derive_soil_parameters!(soil_parameters)
    convert_nijssen2001_to_arno!(soil_parameters)

    return Model(
        ;
        config,
        clock,
        grid_parameters,
        vegetation_parameters,
        soil_parameters,
        surface_energy_variables,
        canopy_variables,
        soil_variables,
        snow_variables,
        forcing_variables,
        forcing_readers,
        routing,
    )
end

function update!(model::Model)
    advance!(model.clock)
    update_forcing!(model.clock.time, model.forcing_readers, model.forcing_variables)

    # Initialize surface temperature on first timestep
    if model.clock.iteration == 1
        model.surface_energy_variables.surface_temperature .= 
        model.forcing_variables.air_temperature
    end

    # Energy balance and atmospheric calculations
    update_energy_balance!(model)

    # Compute canopy evaporation
    update_canopy_evaporation!(model)

    # Transpiration
    update_transpiration!(model)
    
    # Water balance: throughfall (must run BEFORE snow dynamics so
    #   the snow kernel sees today's precipitation, not yesterday's)
    update_water_canopy_storage!(model)

    update_snow!(model)

    update_soil!(model)

    # update total fluxes
    update_total_evapotranspiration!(model)
    update_total_runoff!(model)

    # run routing
    #  Note: fix violation_counter
    update_routing!(model)

    update_soil_conductivity!(model)
    update_soil_volumetric_heat_capacity!(model)
    estimate_soil_layer_temperature!(model)

    if model.clock.iteration == 1
        update_surface_temperature!(model)
        update_aerodynamic_resistance!(model)
    end
    update_surface_temperature!(model)

    update_net_radiation_post_closure!(model)

    # post process
    results = process_daily_outputs!(model)

    # write away results
    return nothing
end


config_file = "/home/bart/git/mGV/configs/mekong_config.toml"
cfg = load_config(config_file)

m = Model(cfg)
update!(m)

t0 = time()
end_time = DateTime(m.config.end_year, 12, 31)
while m.clock.time < end_time
    update!(m)
    println(maximum(m.routing.discharge))
end
print(time() - t0)


end # module end
