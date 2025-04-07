module SimConstants
    export N_veg, K_L, K, z2, Ri_cr, emissivity, 
           Ki, Kw, Kdry_org, Ks_org, 
           organic_frac, bulk_dens_org, soil_dens_org

    const N_veg = 20
    const K_L = 0.2 # As used by Dickinson (1984) [mm]
    const K = 0.4
    const z2 = 10
    const Ri_cr = 0.2
    const emissivity = 0.97 

    # Thermal conductivity constants
    const Ki       = 2.2   # Thermal conductivity of ice (W/mK)
    const Kw       = 0.57  # Thermal conductivity of water (W/mK)
    const Kdry_org = 0.05  # Dry thermal conductivity of organic fraction (W/mK) (Farouki 1981)
    const Ks_org   = 0.25  # Thermal conductivity of organic solid (W/mK) (Farouki 1981)

    # Ground composition constants
    const organic_frac = 0
    const bulk_dens_org = 0
    const soil_dens_org = 0

end