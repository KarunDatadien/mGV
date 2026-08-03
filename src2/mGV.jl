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
using Adapt: adapt, @adapt_structure
using Dates: Second

include("config.jl")
using .Config: load_config, Cfg
include("backend_setup.jl")
include("constants.jl")
using .Constants: PhysConsts, SimConsts, SnowConsts

include("reader.jl")
include("physics.jl")
include("parameters.jl")

config_file = "/home/bart/git/mGV/configs/mekong_config.toml"
cfg = load_config(config_file)

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

"""Initialize clock based on config"""
function Clock(config::Cfg)
    return Clock(
        DateTime(config.start_year),
        0,
        Second(config.timestep)
    )
end


@kwdef struct Model
    config::Cfg    # all configuration options
    clock::Clock   # to keep track of simulation time
    grid_parameters::GridParameters
    vegetation_parameters::VegetationParameters
    soil_parameters::SoilParameters
    surface_energy_variables::SurfaceEnergyVariables
    canopy_variables::CanopyVariables
    soil_variables::SoilVariables
    forcing_variables::ForcingVariables
    forcing_readers::ForcingReaders
    # routing::R                      # routing model (horizontal fluxes), moves along network
    # writer::W                       # writes model output
end

function Model(config::Cfg)
    clock = Clock(config)
    forcing_readers, forcing_variables = initialize_forcing(config)
    grid_parameters, vegetation_parameters, soil_parameters = 
        read_parameters(config)

    # Check dim order!
    nx = length(grid_parameters.longitude)
    ny = length(grid_parameters.latitude)
    nveg = cfg.nveg  # vegetation types
    nbands = cfg.nbands  # snow bands
    nlayers = size(soil_parameters.depth, 3) # derive soil layers from input data
    grid_dims = (nx, ny)
    tile_dims = (nx, ny, nbands, nveg)
    soil_dims = (nx, ny, nlayers)

    surface_energy_variables = SurfaceEnergyVariables(grid_dims)
    canopy_variables = CanopyVariables(tile_dims)
    soil_variables = SoilVariables(soil_dims)

    # Move data to backend during model initialization
    if backend_name != "CPU"
        grid_parameters = adapt(ArrayType, grid_parameters)
        vegetation_parameters = adapt(ArrayType, vegetation_parameters)
        soil_parameters = adapt(ArrayType, soil_parameters)
        surface_energy_variables = adapt(ArrayType, surface_energy_variables)
        canopy_variables = adapt(ArrayType, canopy_variables)
        soil_variables = adapt(ArrayType, soil_variables)
        forcing_variables = adapt(ArrayType, forcing_variables)
    end

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
        forcing_variables,
        forcing_readers,
    )
end

m = Model(cfg)

function update!(model::Model)
    0.0
end

end
