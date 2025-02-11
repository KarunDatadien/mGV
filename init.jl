CASE, start_year_arg, end_year_arg = parse_case_args()
check_and_set_gpu_usage()

# Load the configuration matching the input argument:
if CASE == "global"

    println("Loading configuration for 'global'...")

    # =========================== GLOBAL CONFIGURATION ===========================
        
    # Input file paths/names
    input_param_file       = "./global_data/vic_global_5min_params_fix2.nc"
    input_prec_prefix      = "./global_data/5arcmin/forcing/prec/prec_WFDE5_CRU+GPCC_v2.0_5arcmin_"
    input_tair_prefix      = "./global_data/5arcmin/forcing/tair/tair_WFDE5_v2.0_5arcmin_"
    input_wind_prefix      = "./global_data/5arcmin/forcing/wind/wind_WFDE5_v2.0_5arcmin_"
    input_vp_prefix        = "./global_data/5arcmin/forcing/vp/vp_WFDE5_v2.0_5arcmin_"
    input_swdown_prefix    = "./global_data/5arcmin/forcing/swdown/swdown_WFDE5_v2.0_5arcmin_"
    input_lwdown_prefix    = "./global_data/5arcmin/forcing/lwdown/lwdown_WFDE5_v2.0_5arcmin_"
    
    # Input variable names (as specified in the input files' metadata)
    d0_var = "displacement"
    z0_var = "veg_rough"
    LAI_var = "LAI"
    rmin_var = "rmin"
    rarc_var = "rarc"
    elev_var = "elev"
    albedo_var = "albedo"
    root_var = "root_fract" # root_fract(veg_class, root_zone, lat, lon) ;
    #root_fract_layer1 = root_fract[:, 0, :, :]
    #root_fract_layer2 = root_fract[:, 1, :, :]
    
    # === Field Capacity, Wilting Point, and Critical Moisture ===
    Wcr_var = "Wcr_FRACT" #Wcr_FRACT(nlayer, lat, lon) 
    Wfc_var = "Wfc_FRACT" #Wfc_FRACT(nlayer, lat, lon) 
    Wpwp_var = "Wpwp_FRACT" #Wpwp_FRACT(nlayer, lat, lon) 
    #soil_moisture_critical = Wcr_var * soil_moisture_max
    #field_capacity = Wfc_var * soil_moisture_max
    #wilting_point = Wpwp_var * soil_moisture_max

    coverage_var = "fcanopy" #fcanopy(veg_class, month, lat, lon) # "canopy coverage"

    # === Extract Soil Parameters ===
    depth_var = "depth" #depth(nlayer, lat, lon)
    bulk_dens_var = "bulk_density" #bulk_density(nlayer, lat, lon)
    soil_dens_var = "soil_density" #soil_density(nlayer, lat, lon) 

    # === Calculate Bulk Density, Porosity, and Maximum Soil Moisture ===
    #organic_frac = 0
    #bulk_dens_org = 0
    #soil_dens_org = 0

    #bulk_dens_min = (bulk_density - organic_frac * bulk_dens_org) / (1 - organic_frac)
    #soil_dens_min = (soil_density - organic_frac * soil_dens_org) / (1 - organic_frac)
    #porosity = 1 - bulk_dens_min / soil_dens_min
    #soil_moisture_max = depth * porosity * 1000


    prec_var = "prec"
    tair_var = "tair"
    wind_var = "wind" 
    vp_var = "vp"
    swdown_var = "swdown"
    lwdown_var = "lwdown"

    # Output file paths/names
    output_dir             = "./output/"
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
    input_param_file       = "./indus_data/VIC_params_Mirca_calibrated_Indus.nc"
    input_prec_prefix      = "./indus_data/pr_daily_GFDL-ESM4adj_historical/pr_daily_GFDL-ESM4adj_historical_"
    input_tair_prefix      = "./indus_data/tas_daily_GFDL-ESM4adj_historical/tas_daily_GFDL-ESM4adj_historical_"
    input_wind_prefix      = "./indus_data/wind10_daily_GFDL-ESM4_historical/wind10_daily_GFDL-ESM4_historical_"
    input_vp_prefix        = "./indus_data/vp_daily_GFDL-ESM4_historical/vp_daily_GFDL-ESM4_historical_"
    input_swdown_prefix    = "./indus_data/swdown_daily_GFDL-ESM4adj_historical/swdown_daily_GFDL-ESM4adj_historical_"
    input_lwdown_prefix    = "./indus_data/lwdown_daily_GFDL-ESM4adj_historical/lwdown_daily_GFDL-ESM4adj_historical_"
    # input_psurf_prefix     = "./indus_data/psurf_daily_GFDL-ESM4_historical/psurf_daily_GFDL-ESM4_historical_"
    
    # Input variable names (as specified in the input files' metadata)
    d0_var = "displacement"
    z0_var = "veg_rough"
    LAI_var = "LAI"
    rmin_var = "rmin"
    rarc_var = "rarc"
    elev_var = "elev"
    albedo_var = "albedo"
    root_var = "root_fract" # root_fract(veg_class, root_zone, lat, lon) ;

    #root_fract_layer1 = root_fract[:, 0, :, :]
    #root_fract_layer2 = root_fract[:, 1, :, :]
    
    # === Field Capacity, Wilting Point, and Critical Moisture ===
    Wcr_var = "Wcr_FRACT" #Wcr_FRACT(nlayer, lat, lon) 
    Wfc_var = "Wfc_FRACT" #Wfc_FRACT(nlayer, lat, lon) 
    Wpwp_var = "Wpwp_FRACT" #Wpwp_FRACT(nlayer, lat, lon) 
    #soil_moisture_critical = Wcr_var * soil_moisture_max
    #field_capacity = Wfc_var * soil_moisture_max
    #wilting_point = Wpwp_var * soil_moisture_max

    coverage_var = "fcanopy" #fcanopy(veg_class, month, lat, lon) # "canopy coverage"

    # === Extract Soil Parameters ===
    depth_var = "depth" #depth(nlayer, lat, lon)
    bulk_dens_var = "bulk_density" #bulk_density(nlayer, lat, lon)
    soil_dens_var = "soil_density" #soil_density(nlayer, lat, lon) 

    # === Calculate Bulk Density, Porosity, and Maximum Soil Moisture ===
    #organic_frac = 0
    #bulk_dens_org = 0
    #soil_dens_org = 0

    #bulk_dens_min = (bulk_density - organic_frac * bulk_dens_org) / (1 - organic_frac)
    #soil_dens_min = (soil_density - organic_frac * soil_dens_org) / (1 - organic_frac)
    #porosity = 1 - bulk_dens_min / soil_dens_min
    #soil_moisture_max = depth * porosity * 1000
    
    prec_var = "pr"
    tair_var = "tas"
    wind_var = "wind10" 
    vp_var = "vp"
    swdown_var = "swdown"
    lwdown_var = "lwdown"

    # Output file paths/names
    output_dir             = "./output_indus/"
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

