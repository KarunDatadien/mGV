function solve_surface_temperature(
    Ta::CuArray, T1::CuArray, T2::CuArray, albedo::CuArray,
    Rs::CuArray, RL::CuArray, rh::CuArray, kappa::CuArray,
    D1, D2, delta_t, Cs::CuArray, E_n::CuArray
)
    # === Constants (assumed globally defined in Main) ===
    @assert isdefined(Main, :lat_vap) "lat_vap is undefined"
    @assert isdefined(Main, :c_p_air) "c_p_air is undefined"
    @assert isdefined(Main, :emissivity) "emissivity is undefined"
    @assert isdefined(Main, :sigma) "sigma is undefined"
    @assert isdefined(Main, :p_std) "p_std is undefined"
    
    # Debugging prints (synchronize GPU before printing)
    println("Cs has NaN: ", any(isnan, Cs), " min/max: ", minimum(Cs), " / ", maximum(Cs))
    println("kappa has NaN: ", any(isnan, kappa), " min/max: ", minimum(kappa), " / ", maximum(kappa))
    println("rh has NaN: ", any(isnan, rh), " min/max: ", minimum(rh), " / ", maximum(rh))

    # === Compute dependent quantities ===
    latent_heat = calculate_latent_heat(Ta) # should be with dimension W*m^-2
    rho_a = 1.225 # Density of air (TODO: make temperature dependent?)

    println("latent_heat min/max: ", minimum(latent_heat), " / ", maximum(latent_heat))

    # === Constants ===
    top_layer_depth = 0.3 # TODO: good value?
    rho_w = 1000.0  # Density of liquid water (TODO: make temperature dependent?)

    # === Precompute constants ===
    base_term = (kappa ./ D2) .+ (Cs .* D2 ./ (2 * delta_t))
    heat_transfer_term = base_term ./ (1 .+ (D1 / D2) .+ (Cs .* D1 .* D2 ./ (2 * delta_t .* kappa)))
    air_term = (rho_a .* c_p_air ./ rh) .+ (rho_a .* c_p_air .* top_layer_depth ./ (2 * delta_t))
    common_term = heat_transfer_term .+ air_term

    println("base_term min/max: ", minimum(base_term), " / ", maximum(base_term))
    println("heat_transfer_term min/max: ", minimum(heat_transfer_term), " / ", maximum(heat_transfer_term))
    println("air_term min/max: ", minimum(air_term), " / ", maximum(air_term))
    println("common_term min/max: ", minimum(common_term), " / ", maximum(common_term))

    # === Define the residual function f(Ts_new) = lhs - rhs ===
    function f(Ts_new, Ts_old)
        lhs = emissivity .* sigma .* Ts_new.^4 .+ common_term .* Ts_new
        rhs = (1 .- albedo) .* Rs .+ emissivity .* RL .+
              (rho_a .* c_p_air ./ rh) .* Ta .- rho_w .* lat_vap .* (E_n ./ (day_sec .* mm_in_m)) .+ # E_n: convert mm/day to m/s
              (rho_a .* c_p_air .* top_layer_depth .* Ts_old ./ (2 * delta_t)) .+
              ((kappa .* T2 ./ D2) .+ (Cs .* D2 .* T1 ./ (2 * delta_t))) ./ 
              (1 .+ (D1 / D2) .+ (Cs .* D1 .* D2 ./ (2 * delta_t .* kappa)))
        return lhs .- rhs
    end

    # === Derivative of f(Ts_new) for Newton-Raphson ===
    function df_dTs_new(Ts_new, Ts_old)
        # Derivative w.r.t. Ts_new: d/dTs_new (emissivity * sigma * Ts_new^4 + common_term * Ts_new)
        # = 4 * emissivity * sigma * Ts_new^3 + common_term
        return 4 .* emissivity .* sigma .* Ts_new.^3 .+ common_term
    end

    # === Newton-Raphson solver ===
    Ts_old = copy(Ta)  # Initial guess on GPU
    Ts_new = copy(Ts_old)
    tolerance = 1e-6
    max_iter = 10

    for iter in 1:max_iter
        residual = f(Ts_new, Ts_old)
        derivative = df_dTs_new(Ts_new, Ts_old)
        
        # Newton step: Ts_new = Ts_new - f(Ts_new) / f'(Ts_new)
        delta_Ts = residual ./ derivative
        Ts_new = Ts_new .- delta_Ts
        
        # Check convergence
        max_delta = maximum(abs.(delta_Ts))
        min_delta = minimum(abs.(delta_Ts))

        println("Iteration $iter: Max delta Ts = $max_delta")
        println("Iteration $iter: Min delta Ts = $min_delta")
        if max_delta < tolerance
            println("Converged after $iter iterations")
            break
        end
        
        # Update Ts_old for next iteration
        Ts_old = Ts_new
    end

    return Ts_new
end

