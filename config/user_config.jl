CASE, start_year_arg, end_year_arg = parse_case_args()
check_and_set_gpu_usage()

# Load the configuration matching the input argument:
if CASE == "global"

    println("Loading configuration for 'global'...")

    # =========================== GLOBAL CONFIGURATION ===========================
        
    # Input file paths/names
    input_param_file       = "./input_data/global/vic_global_5min_params_fix2.nc"
    input_prec_prefix      = "./input_data/global/5arcmin/forcing/prec/prec_WFDE5_CRU+GPCC_v2.0_5arcmin_"
    input_tair_prefix      = "./input_data/global/5arcmin/forcing/tair/tair_WFDE5_v2.0_5arcmin_"
    input_wind_prefix      = "./input_data/global/5arcmin/forcing/wind/wind_WFDE5_v2.0_5arcmin_"
    input_vp_prefix        = "./input_data/global/5arcmin/forcing/vp/vp_WFDE5_v2.0_5arcmin_"
    input_swdown_prefix    = "./input_data/global/5arcmin/forcing/swdown/swdown_WFDE5_v2.0_5arcmin_"
    input_lwdown_prefix    = "./input_data/global/5arcmin/forcing/lwdown/lwdown_WFDE5_v2.0_5arcmin_"
    
    # Input variable names (as specified in the input files' metadata)
    d0_var = "displacement"
    z0_var = "veg_rough"
    LAI_var = "LAI"
    rmin_var = "rmin"
    rarc_var = "rarc"
    elev_var = "elev"
    residmoist_var = "resid_moist"
    init_moist_var = "init_moist"

    ksat_var = "Ksat"
    albedo_var = "albedo"
    root_var = "root_fract" # root_fract(veg_class, root_zone, lat, lon) ;
    #root_fract_layer1 = root_fract[:, 0, :, :]
    #root_fract_layer2 = root_fract[:, 1, :, :]
    
    # === Field Capacity, Wilting Point, and Critical Moisture related variables ===
    Wcr_var = "Wcr_FRACT" #Wcr_FRACT(nlayer, lat, lon) 
    Wfc_var = "Wfc_FRACT" #Wfc_FRACT(nlayer, lat, lon) 
    Wpwp_var = "Wpwp_FRACT" #Wpwp_FRACT(nlayer, lat, lon) 
    coverage_var = "fcanopy" #fcanopy(veg_class, month, lat, lon) # "canopy coverage"
    quartz_var = "quartz" #quartz(nlayer, lat, lon)

    # === Extract Soil Parameters ===
    depth_var = "depth" #depth(nlayer, lat, lon)
    bulk_dens_var = "bulk_density" #bulk_density(nlayer, lat, lon)
    soil_dens_var = "soil_density" #soil_density(nlayer, lat, lon) 
    expt_var = "expt"
    b_infilt_var = "infilt"

    # === Subsurface Parameters ===
    Ds_var = "Ds" #fraction
    Dsmax_var = "Dsmax" #mm/day
    Ws_var = "Ws" #fraction

    prec_var = "prec"
    tair_var = "tair"
    wind_var = "wind" 
    vp_var = "vp"
    swdown_var = "swdown"
    lwdown_var = "lwdown"

    # Output file paths/names
    output_dir             = "./output_data/global/"
    output_file_prefix     = "outputfile_global_"
    
    # Set default simulation years if no command-line arguments are provided
    start_year             = isnothing(start_year_arg) ? 1979 : start_year_arg
    end_year               = isnothing(end_year_arg)   ? 2019 : end_year_arg
    
    # ========================= END GLOBAL CONFIGURATION ==========================

    ensure_output_directory(output_dir)
    println("Running from year $start_year to year $end_year.\n")

elseif CASE == "indus"

    println("Loading configuration for 'indus'...")
    
    # ============================ INDUS CONFIGURATION ============================
   
    # Input file paths/names
    input_param_file       = "./input_data/indus/VIC_params_Mirca_calibrated_Indus.nc"
    input_prec_prefix      = "./input_data/indus/forcing/pr_daily_GFDL-ESM4adj_historical/pr_daily_GFDL-ESM4adj_historical_"
    input_tair_prefix      = "./input_data/indus/forcing/tas_daily_GFDL-ESM4adj_historical/tas_daily_GFDL-ESM4adj_historical_"
    input_wind_prefix      = "./input_data/indus/forcing/wind10_daily_GFDL-ESM4_historical/wind10_daily_GFDL-ESM4_historical_"
    input_vp_prefix        = "./input_data/indus/forcing/vp_daily_GFDL-ESM4_historical/vp_daily_GFDL-ESM4_historical_"
    input_swdown_prefix    = "./input_data/indus/forcing/swdown_daily_GFDL-ESM4adj_historical/swdown_daily_GFDL-ESM4adj_historical_"
    input_lwdown_prefix    = "./input_data/indus/forcing/lwdown_daily_GFDL-ESM4adj_historical/lwdown_daily_GFDL-ESM4adj_historical_"
    # input_psurf_prefix     = "./input_data/indus/forcing/psurf_daily_GFDL-ESM4_historical/psurf_daily_GFDL-ESM4_historical_"
    
    # Input variable names (as specified in the input files' metadata)
    d0_var = "displacement"
    z0_var = "veg_rough"
    LAI_var = "LAI"
    rmin_var = "rmin"
    rarc_var = "rarc"
    elev_var = "elev"
    residmoist_var = "resid_moist"
    init_moist_var = "init_moist"

    ksat_var = "Ksat"
    albedo_var = "albedo"
    root_var = "root_fract" # root_fract(veg_class, root_zone, lat, lon) ;
    #root_fract_layer1 = root_fract[:, 0, :, :]
    #root_fract_layer2 = root_fract[:, 1, :, :]
    
    # === Field Capacity, Wilting Point, and Critical Moisture related variables ===
    Wcr_var = "Wcr_FRACT" #Wcr_FRACT(nlayer, lat, lon) 
    Wfc_var = "Wfc_FRACT" #Wfc_FRACT(nlayer, lat, lon) 
    Wpwp_var = "Wpwp_FRACT" #Wpwp_FRACT(nlayer, lat, lon) 
    coverage_var = "fcanopy" #fcanopy(veg_class, month, lat, lon) # "canopy coverage"
    quartz_var = "quartz" #quartz(nlayer, lat, lon)

    # === Extract Soil Parameters ===
    depth_var = "depth" #depth(nlayer, lat, lon)
    bulk_dens_var = "bulk_density" #bulk_density(nlayer, lat, lon)
    soil_dens_var = "soil_density" #soil_density(nlayer, lat, lon) 
    expt_var = "expt"
    b_infilt_var = "infilt"

    # === Subsurface Parameters ===
    Ds_var = "Ds" #fraction
    Dsmax_var = "Dsmax" #mm/day
    Ws_var = "Ws" #fraction

    prec_var = "pr"
    tair_var = "tas"
    wind_var = "wind10" 
    vp_var = "vp"
    swdown_var = "swdown"
    lwdown_var = "lwdown"

    # Output file paths/names
    output_dir             = "./output_data/indus/"
    output_file_prefix     = "outputfile_indus_"
    
    # Set default simulation years if no command-line arguments are provided
    start_year             = isnothing(start_year_arg) ? 1979 : start_year_arg
    end_year               = isnothing(end_year_arg)   ? 2010 : end_year_arg
    
    # ========================== END INDUS CONFIGURATION ==========================

    ensure_output_directory(output_dir)
    println("Running from year $start_year to year $end_year. \n")

else
    error("Unknown CASE: '$CASE'. Please provide 'global' or 'indus' (or any other case defined in init.jl) as the first argument.")
end

