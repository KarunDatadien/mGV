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

include("physics.jl")
include("config.jl")
using .Config: load_config, Cfg
include("constants.jl")
using .Constants: PhysConsts, SimConsts, SnowConsts

include("parameters.jl")


config_file = "/home/bart/git/mGV/configs/mekong_config.toml"
cfg = load_config(config_file)

struct Model{T}
    config::Cfg                  # all configuration options
    grid_parameters::GridParameters
    vegetation_parameters::VegetationParameters
    snow_parameters::SoilParameters
    surface_energy_variables::SurfaceEnergyVariables
    canopy_variables::CanopyVariables
    soil_variables::SoilVariables
    # routing::R                      # routing model (horizontal fluxes), moves along network
    # reader::NCReader                # provides the model with dynamic input
    # writer::W                       # writes model output
    # consider adding; clock::Clock                    # to keep track of simulation time
end

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