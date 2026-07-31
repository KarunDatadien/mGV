using NCDatasets.CommonDataModel: CFVariable, @select
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
    time::CFVariable
    latitude::CFVariable
    longitude::CFVariable
    precipitation::CFVariable
    air_temperature::CFVariable
    wind_speed::CFVariable
    vapor_pressure::CFVariable
    shortwave_down::CFVariable
    longwave_down::CFVariable
    surface_pressure::CFVariable
end

"""
Open the forcing input data files to prepare for stepwise data
loading.
"""
function open_forcing(cfg::Cfg)
    years = cfg.start_year:cfg.end_year
    var_prefixes = [getval(cfg.input.paths, "$(var)_file") for var in FORCING_VARS]
    files = Vector{String}(undef, length(years) * length(var_prefixes))

    for i = eachindex(var_prefixes)
        for j = eachindex(years)
            ncfile = validate_path(
                "$(var_prefixes[i])$(years[j]).nc",
                dirname(config_file)
            )
            files[j+length(years)*(i-1)] = ncfile
        end
    end

    files = unique(files)  # vars can be in the same files, eg 1 zarr store
    ds = NCDataset(files, aggdim = "")

    vars_dict = Dict(
        var => ds[getval(cfg.input.names, var)]
        for var in [FORCING_COORDS; FORCING_VARS]
    )
    return ForcingReaders(; (Symbol.(keys(vars_dict)) .=> values(vars_dict))...)
end

"""
Read in forcing data at a certain point in time.
"""
function read_var(time, readers::ForcingReaders, var::String)
    var_reader = getval(readers, var)
    # dim_order = dimnames(var_reader)  # use to check dim order with alternative input data
    time_slice = @select(var_reader, time ≈ $time)
    return nomissing(time_slice, NaN)[:,:]
end

"""
Initialize forcing reader and read the first timestep of
forcing data.
"""
function initialize_forcing(cfg::Cfg)
    forcing_readers = open_forcing(cfg)
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
