using .PhysicalConstants

# Vapor Pressure Deficit Calculation
function calculate_vpd(tair::CuArray{Float32}, vp::CuArray{Float32})
    svp = svp_a .* exp.((svp_b .* tair) ./ (svp_c .+ tair))
    svp = ifelse.(tair .< 0, svp .* (1.0 .+ 0.00972 .* tair .+ 0.000042 .* tair.^2), svp)
    vpd = svp .- vp
    return vpd .* pa_per_kpa
end

# Saturation Vapor Pressure Slope
function calculate_svp_slope(temp::CuArray{Float32})
    return (svp_b .* svp_c) ./ ((svp_c .+ temp).^2) .* (svp_a .* exp.((svp_b .* temp) ./ (svp_c .+ temp)))
end

# Scale Height Calculation
function calculate_scale_height(tair_gpu, elev)
    return r_air ./ g .* ((tair_gpu .+ t_freeze) .+ 0.5 .* elev .* lapse_rate)
end

# Latent Heat of Vaporization
function calculate_latent_heat(temp::CuArray{Float32})
    return lat_vap .- 2361.0 .* temp
end
