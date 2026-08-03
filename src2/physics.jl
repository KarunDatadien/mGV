struct SurfaceEnergyVariables{T <: AbstractMatrix}
    # State
    surface_temperature::T

    # Fluxes
    net_radiation::T
    potential_evaporation::T
    soil_potential_evaporation::T
    total_evapotranspiration::T

    # Derived/intermediate
    aerodynamic_resistance::T
end

@adapt_structure SurfaceEnergyVariables

function SurfaceEnergyVariables(dims)
    return SurfaceEnergyVariables(
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims)
    )
end

struct CanopyVariables{T <: AbstractArray}
    # State
    water_storage::T

    # Fluxes
    throughfall::T
    canopy_evaporation::T
    transpiration::T
    transpiration_layers::T

    # Derived/intermediate
    maximum_water_storage::T
    wet_fraction::T
end

@adapt_structure CanopyVariables

function CanopyVariables(dims)
    return CanopyVariables(
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims)
    )
end


struct SoilVariables{T <: AbstractArray}
    # State
    moisture::T
    temperature::T
    ice_fraction::T

    # Fluxes
    evaporation::T
    infiltration::T
    surface_runoff::T
    subsurface_runoff::T
    total_runoff::T
    interlayer_drainage::T

    # Derived/intermediate
    thermal_conductivity::T
    heat_capacity::T
    saturated_fraction::T
end

@adapt_structure SoilVariables

function SoilVariables(dims)
    return SoilVariables(
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims)
    )
end

;SVP_A, SVP_B, SVP_C, PA_PER_KPA, R_AIR, T_FREEZE, LAPSE_RATE = PhysConsts

"""
    calculate_svp(air_temperature)

Compute the saturation vapor pressure (kPa?) based on the air
temperature (°C).
"""
@inline function calculate_svp(air_temperature)
    # 1. Tetens Equation (standard over water)
    svp = SVP_A * exp((SVP_B * air_temperature) / (SVP_C + air_temperature))

    # 2. Sub-zero correction (Murray 1967 / standard VIC logic)
    # Lower saturation vapor pressure over ice compared to water
    if air_temperature < 0.0f0
        svp = svp * (1.0f0 + 0.00972f0 * air_temperature + 0.000042f0 * air_temperature^2)
    end
    return svp
end

"""
    calculate_vpd(air_temperature, vapor_pressure)

Compute the vapor pressure deficit (Pa?) based on the air temperature (°C)
and actual vapor pressure (kPa?).
"""
function calculate_vpd(air_temperature, vapor_pressure)
    return max(
        svp(air_temperature) - vapor_pressure,
        0.0f0
    ) * PA_PER_KPA # [Pa]
end

"""
Compute the slope of the saturation vapor pressure curve.
"""
@inline function calculate_svp_slope(air_temperature)
    # Re-calculate SVP part locally (scalar)
    svp_part = SVP_A * exp((SVP_B * air_temperature) / (SVP_C + air_temperature))
    
    # Calculate Slope
    slope_kpa = (SVP_B * SVP_C * svp_part) / ((SVP_C + air_temperature)^2)
    
    return slope_kpa * PA_PER_KPA # [Pa/°C]
end

"""
Compute the atmospheric scale height [m] with lapse rate correction
"""
@inline function calculate_scale_height(air_temperature, elevation)
    scale_height = (R_AIR / G) * (
        (air_temperature + T_FREEZE) + 0.5f0 * elevation * LAPSE_RATE
    )
    return scale_height
end

"""
Compute the latent heat of vaporization (J/kg)
"""
@inline function calculate_latent_heat(air_temperature_kelvin)
    tc = air_temperature_kelvin - T_FREEZE
    return 2.501f6 - 2361.0f0 * tc # [K/kg]
end
