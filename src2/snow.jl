struct SnowVariables{T <: AbstractArray, I <: AbstractArray, V <: AbstractArray}
    # 4D (nx, ny, nbands, nveg)
    snow_water_equivalent::T
    surface_water::T
    snowpack_water::T
    depth::T
    albedo::T
    surface_temperature::T
    coverage::T
    melt::T
    last_snow::I          # days since last snowfall
    cold_content::T       # J/m² surface layer cold content
    pack_cold_content::T  # J/m² pack layer cold content (2-layer)
    melting_flag::I       # melt-season flag
    # coverage
    store::I                     # coverage state (1 = store)
    depth_distribution_slope::T  # (m)
    stored_swe::T                # stored SWE for coverage (mm)
    stored_coverage::T           # stored coverage fraction
    max_snow_depth::T            # max depth for coverage (m)
 
    #3D (aggregations across veg tiles, for soil input)
    # (nx, ny, nbands)
    aggregated_melt::V
    aggregated_rain::V
    infiltration::V  # precipitation/melt reaching soil surface (old name: PPT)

    # Forcings
    band_air_temperature::V
    band_precipitation::V
end

@adapt_structure SnowVariables

function SnowVariables(nx, ny, bands, nveg)
    snow_dims = (nx, ny, bands, nveg)
    band_dims = (nx, ny, bands)  # aggregations over vegetation tiles

    return SnowVariables(
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Int32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Int32, snow_dims),
        zeros(Int32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, snow_dims),
        zeros(Float32, band_dims),
        zeros(Float32, band_dims),
        zeros(Float32, band_dims),
        zeros(Float32, band_dims),
        zeros(Float32, band_dims),
    )
end
