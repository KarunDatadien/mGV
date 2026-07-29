struct SurfaceEnergyVariables{T <: AbstractArray}
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

