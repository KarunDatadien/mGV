using CUDA
using LinearAlgebra  # For mathematical operations

module PhysicalConstants
    export svp_a, svp_b, svp_c, pa_per_kpa, 
           k_b, n_a, r_gas, mw_air, r_air, 
           t_freeze, lapse_rate, lat_vap, g, sigma, 
           p_std, c_p_air, day_sec

    # Saturation Vapor Pressure Constants
    const svp_a = 0.61078
    const svp_b = 17.269
    const svp_c = 237.3
    const pa_per_kpa = 1000

    # Universal Physical Constants
    const k_b = 1.38065e-23  # Boltzmann's constant (J/K)
    const n_a = 6.02214e26   # Avogadro's number (molecules/kmole)
    const r_gas = n_a * k_b  # Universal gas constant (J/K/kmole)
    const mw_air = 28.966    # Molecular weight of dry air (kg/kmole)
    const r_air = r_gas / mw_air  # Dry air gas constant (J/K/kg)

    # Temperature and Environmental Constants
    const t_freeze = 273.15   # Freezing temperature (K)
    const lapse_rate = -0.0065  # Lapse rate (K/m)

    # Energy and Radiation Constants
    const lat_vap = 2.501e6   # Latent heat of vaporization (J/kg)
    const g = 9.81            # Gravitational acceleration (m/s²)
    const sigma = 5.67e-8     # Stefan-Boltzmann constant (W/m²K⁴)

    # Atmospheric Constants
    const p_std = 101325.0    # Standard pressure (Pa)
    const c_p_air = 1013      # Specific heat of moist air (J/kg·K)
    const day_sec = 86400     # Seconds in a day
end

# Submodule for GPU-Accelerated Functions
module PhysicsFunctions

    # Vapor Pressure Deficit Calculation
    function calculate_vpd_gpu(tair::CuArray{Float32}, vp::CuArray{Float32})
        svp = svp_a .* exp.((svp_b .* tair) ./ (svp_c .+ tair))
        svp = ifelse.(tair .< 0, svp .* (1.0 .+ 0.00972 .* tair .+ 0.000042 .* tair.^2), svp)
        vpd = svp .- vp
        return vpd .* pa_per_kpa
    end

    # Saturation Vapor Pressure Slope
    function calculate_svp_slope_gpu(temp::CuArray{Float32})
        return (svp_b * svp_c) ./ ((svp_c .+ temp).^2) .* (svp_a .* exp.((svp_b .* temp) ./ (svp_c .+ temp)))
    end

    # Scale Height Calculation
    function calculate_scale_height_gpu(tair::CuArray{Float32}, elevation::CuArray{Float32})
        return r_air / g .* ((tair .+ t_freeze) .+ 0.5 .* elevation .* lapse_rate)
    end

    # Latent Heat of Vaporization
    function calculate_latent_heat_gpu(temp::CuArray{Float32})
        return lat_vap .- 2361.0 .* temp
    end
end

