using .PhysicalConstants

# Vapor Pressure Deficit Calculation
function calculate_vpd(tair, vp)
    svp = svp_a .* exp.((svp_b .* tair) ./ (svp_c .+ tair)) # Tetens equation
    svp = ifelse.(tair .< 0, svp .* (1.0 .+ 0.00972 .* tair .+ 0.000042 .* tair.^2), svp) # Sub-zero correction (lower vp over ice vs. water)
    vpd = svp .- vp
    return vpd .* pa_per_kpa # Pa
end

# Saturation Vapor Pressure Slope
function calculate_svp_slope(temp)
    return (svp_b .* svp_c) ./ ((svp_c .+ temp).^2) .* (svp_a .* exp.((svp_b .* temp) ./ (svp_c .+ temp))) .* pa_per_kpa # [Pa/°C]
end

# Scale Height Calculation
function calculate_scale_height(tair_gpu, elev)
    return r_air ./ g .* ((tair_gpu .+ t_freeze) .+ 0.5 .* elev .* lapse_rate) # [m]
end

# Latent Heat of Vaporization
#function calculate_latent_heat(temp)
#    return lat_vap .- 2361.0 .* temp # [J/kg]
#end

# Watson correlation
function calculate_latent_heat(T)
    # Parameters for water
    Hvap_Tb = 2.26e6  # Latent heat at boiling point (J/kg)
    Tb = 373.15       # Boiling point (K)
    Tc = 647.096      # Critical temperature (K)
    n = 0.38          # Watson exponent

    # Compute element-wise ratio
    ratio = (Tc .- T) ./ (Tc - Tb)
    ratio = clamp.(ratio, 1e-6, 1.0)  # Prevent unphysical values
    Hvap = Hvap_Tb .* (ratio .^ n)

    # Ensure latent heat is positive, element-wise
    return max.(Hvap, 1e3)  # Minimum 1000 J/kg for numerical stability
end