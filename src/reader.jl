using NCDatasets.CommonDataModel: CFVariable, MFCFVariable, @select
using Dates: DateTime

const FORCING_COORDS = [
    "latitude",
    "longitude",
    "time",
]

const FORCING_VARS = [
    "precipitation",
    "air_temperature",
    "wind_speed",
    "vapor_pressure",
    "shortwave_down",
    "longwave_down",
    "surface_pressure"
]

function getval(data::Any, name::String)
    return getfield(data, Symbol(name))
end

@kwdef struct ForcingVariables{T <: AbstractMatrix}
    precipitation::T
    air_temperature::T
    wind_speed::T
    vapor_pressure::T
    shortwave_down::T
    longwave_down::T
    surface_pressure::T
end

@adapt_structure ForcingVariables

@kwdef struct ForcingReaders
    time::Union{CFVariable, MFCFVariable}
    latitude::Union{CFVariable, MFCFVariable}
    longitude::Union{CFVariable, MFCFVariable}
    precipitation::Union{CFVariable, MFCFVariable}
    air_temperature::Union{CFVariable, MFCFVariable}
    wind_speed::Union{CFVariable, MFCFVariable}
    vapor_pressure::Union{CFVariable, MFCFVariable}
    shortwave_down::Union{CFVariable, MFCFVariable}
    longwave_down::Union{CFVariable, MFCFVariable}
    surface_pressure::Union{CFVariable, MFCFVariable}
end

"""
Open the forcing input data files to prepare for stepwise data
loading.
"""
function open_forcing(config_file::AbstractString, cfg::Cfg)
    years = cfg.start_year:cfg.end_year
    var_prefixes = [getval(cfg.input.paths, "$(var)_file") for var in FORCING_VARS]
    files = Vector{String}(undef, length(years))
    datasets = Vector{Any}(undef,  length(var_prefixes))

    for i = eachindex(var_prefixes)
        for j = eachindex(years)
            ncfile = validate_path(
                "$(var_prefixes[i])$(years[j]).nc",
                dirname(config_file)
            )
            files[j] = ncfile
        end
        datasets[i] = NCDataset(unique(files), aggdim = "time", deferopen = false)
    end

    vars_dict = Dict()

    for i in eachindex(FORCING_VARS)
        var = FORCING_VARS[i]
        vars_dict[var] = datasets[i][getval(cfg.input.names, var)]
    end
    for var in FORCING_COORDS
        vars_dict[var] = datasets[1][getval(cfg.input.names, var)]
    end

    return ForcingReaders(; (Symbol.(keys(vars_dict)) .=> values(vars_dict))...)
end

"""
Read in forcing data at a certain point in time.
"""
function read_var(time, readers::ForcingReaders, var::String)
    var_reader = getval(readers, var)
    # dim_order = dimnames(var_reader)  # use to check dim order with alternative input data
    time_slice = @select(var_reader, time ≈ $time)
    return time_slice.var[:,:] :: Array{Float32, 2}  # NOTE: no missing data checks
end

"""
Initialize forcing reader and read the first timestep of
forcing data.
"""
function initialize_forcing(config_file::AbstractString, cfg::Cfg)
    forcing_readers = open_forcing(config_file, cfg)
    start_time = DateTime(cfg.start_year,1,1)
    forcing_vars = ForcingVariables(;
        ((Symbol(var) => read_var(start_time, forcing_readers, var)) for var in FORCING_VARS)...
    )
    return forcing_readers, forcing_vars
end

"""
Update the forcing variables to the data representing a new time step.

Note: will update CPU as well as GPU arrays in-place.
"""
function update_forcing!(time, readers::ForcingReaders, vars::ForcingVariables)
    @views begin
        vars.precipitation[:] = read_var(time, readers, "precipitation")
        vars.air_temperature[:] = read_var(time, readers, "air_temperature")
        vars.wind_speed[:] = read_var(time, readers, "wind_speed")
        vars.vapor_pressure[:] = read_var(time, readers, "vapor_pressure")
        vars.shortwave_down[:] = read_var(time, readers, "shortwave_down")
        vars.longwave_down[:] = read_var(time, readers, "longwave_down")
        vars.surface_pressure[:] = read_var(time, readers, "surface_pressure")
    end
    return nothing
end
