function solve_surface_temperature(
    Ta::CuArray, T1::CuArray, T2::CuArray, albedo::CuArray,
    Rs::CuArray, RL::CuArray, rh::CuArray, kappa::CuArray,
    D1, D2, delta_t, elev_gpu::CuArray,
    vp_gpu::CuArray, Cs::CuArray, E_n::CuArray
)

    # === Constants (check if these are globally defined!) ===
    @assert isdefined(Main, :lat_vap) "lat_vap is undefined"
    @assert isdefined(Main, :c_p_air) "c_p_air is undefined"
    @assert isdefined(Main, :emissivity) "emissivity is undefined"
    @assert isdefined(Main, :sigma) "sigma is undefined"
    @assert isdefined(Main, :p_std) "p_std is undefined"
    println("Cs has NaN: ", any(isnan, Cs), " min/max: ", minimum(Cs), " / ", maximum(Cs))
    println("kappa has NaN: ", any(isnan, kappa), " min/max: ", minimum(kappa), " / ", maximum(kappa))
    println("rh has NaN: ", any(isnan, rh), " min/max: ", minimum(rh), " / ", maximum(rh))
    
    Ts = copy(Ta)  # Initial guess

    println("Initial Ts min/max: ", minimum(Ts), " / ", maximum(Ts))

    # === Compute dependent quantities ===
    scale_height = calculate_scale_height(Ta, elev_gpu)
    surface_pressure = p_std .* exp.(-elev_gpu ./ scale_height)
    latent_heat = calculate_latent_heat(Ta)
    rho_a = 0.003486 .* surface_pressure ./ (275 .+ Ta)

    println("rho_a min/max: ", minimum(rho_a), " / ", maximum(rho_a))
    println("latent_heat min/max: ", minimum(latent_heat), " / ", maximum(latent_heat))


    # === Precompute constants ===
    base_term = (kappa ./ D2) .+ (Cs .* D2 ./ (2 * delta_t))
    heat_transfer_term = base_term ./ (1 .+ (D1 / D2) .+ (Cs .* D1 .* D2 ./ (2 * delta_t .* kappa)))
    air_term = rho_a .* c_p_air ./ rh
    common_term = heat_transfer_term .+ air_term

    println("base_term min/max: ", minimum(base_term), " / ", maximum(base_term))
    println("heat_transfer_term min/max: ", minimum(heat_transfer_term), " / ", maximum(heat_transfer_term))
    println("air_term min/max: ", minimum(air_term), " / ", maximum(air_term))
    println("common_term min/max: ", minimum(common_term), " / ", maximum(common_term))

    for iter in 1:5
        lhs = emissivity .* sigma .* Ts.^4 .+ common_term .* Ts

        rhs = (1 .- albedo) .* Rs .+ emissivity .* RL .+
              air_term .* Ta .- rho_a .* lat_vap .* E_n .+
              ((kappa .* T2 ./ D2) .+ (Cs .* D2 .* T1 ./ (2 * delta_t))) ./ 
              (1 .+ (D1 / D2) .+ (Cs .* D1 .* D2 ./ (2 * delta_t .* kappa)))

        f = lhs .- rhs
        df = 4 .* emissivity .* sigma .* Ts.^3 .+ common_term

        Ts_new = Ts .- f ./ df
        if maximum(abs.(Ts_new .- Ts)) < 1e-3
            break
        end

        Ts = Ts_new
    end

    return Ts
end
