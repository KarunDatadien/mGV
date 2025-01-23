using .PhysicalConstants

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

function compute_richardson_number(z2, d0, tsurf, tair, wind)
    """
    Computes the Richardson number for given atmospheric parameters.

    Arguments:
        z      :: Height above ground level (scalar or array)
        d0     :: Zero-plane displacement height (array)
        tsurf  :: Surface temperature (scalar)
        tair   :: Air temperature (array)
        wind   :: Wind speed (array)
        z0     :: Roughness length (array)

    Returns:
        Ri_B     :: The Richardson number (array), clamped to the range [-0.5, 0.2].
    """
    # Calculate Richardson number
    Ri_B = ifelse.(
        tsurf .!= tair,  # Element-wise comparison
        g .* (tair .- tsurf) .* (z2 .- d0) ./ 
        (((tair .+ t_freeze) .+ (tsurf .+ t_freeze)) ./ 2 .* wind.^2),
        0.0
    )

    # Clamp Richardson number to the valid range
    return clamp.(Ri_B, -0.5, Ri_cr)
end