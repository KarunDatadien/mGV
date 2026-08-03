"""
Calculate per-snow-band forcings.
"""
function calculate_band_forcings!(
    grid_parameters::GridParameters,
    forcing_variables::ForcingVariables,
    snow_variables::SnowVariables
)
    (; elevation, snow_band_elevation, snow_band_area_fraction, snow_band_precipitation_factor) = grid_parameters
    (; air_temperature, precipitation) = forcing_variables
    (; band_air_temperature, band_precipitation) = snow_variables
    
    @. band_air_temperature = air_temperature - 0.0065f0 * (snow_band_elevation - elevation)

    # can't the result of this if-else statement be pre-calculated?
    @. band_precipitation = precipitation * ifelse(
                    snow_band_area_fraction > 1f-6,
                    snow_band_precipitation_factor / snow_band_area_fraction,
                    0.0f0
    )
    return nothing
end

"""
Compute the net radiation (ignoring snow).
"""
function calculate_net_radiation!(
    forcing_variables::ForcingVariables,
    surface_energy_variables::SurfaceEnergyVariables,
    albedo
)
    (; shortwave_down, longwave_down) = forcing_variables 
    (; surface_temperature, net_radiation) = surface_energy_variables

    @. net_radiation = (
        (1.0f0 - albedo) * shortwave_down
        + longwave_down
        - Constants.EMISSIVITY * Constants.SIGMA * (surface_temperature + 273.15f0) ^ 4
    )
    return nothing
end

function calculate_net_radiation!(
    forcing_variables::ForcingVariables,
    surface_energy_variables::SurfaceEnergyVariables,
    vegetation_parameters::VegetationParameters,
    snow_variables::SnowVariables,
    snow_parameters
)
    nothing
end
