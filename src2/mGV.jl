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

vars = open_forcing(cfg)

"""
Clock struct for timekeeping.

Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
mutable struct Clock{T}
    time::T
    iteration::Int
    dt::Second
end


struct Model{T}
    config::Cfg    # all configuration options
    clock::Clock   # to keep track of simulation time
    grid_parameters::GridParameters
    vegetation_parameters::VegetationParameters
    snow_parameters::SoilParameters
    surface_energy_variables::SurfaceEnergyVariables
    canopy_variables::CanopyVariables
    soil_variables::SoilVariables
    forcing_variables::ForcingVariables
    forcing_readers::ForcingReaders
    # routing::R                      # routing model (horizontal fluxes), moves along network
    # writer::W                       # writes model output
end

f = initialize_forcing(cfg)

# function Model(config::Cfg)
#     return Model(
#         ;
#         config,
#         grid_parameters,
#         vegetation_parameters,
#         snow_parameters
#         surface_energy_variables,
#         soil_variables
#     )
# end

# function initialize(::Type{Model}, config_file::AbstractString)
#     config = load_config(config_file)
#     model = Model(config)
#     return model
# end

# function update!(model::Model)
#     # ...
# end

end
