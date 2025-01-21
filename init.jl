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

    println("Running from year $start_year to year $end_year.\n")

elseif CASE == "indus"

    println("Loading configuration for 'indus'...")
    
    # ============================ INDUS CONFIGURATION ============================
   
    # Input file paths/names
    input_param_file       = "./indus_data/domain_Indus.nc"
    input_prec_prefix      = "./indus_data/pr_daily_GFDL-ESM4adj_historical/pr_daily_GFDL-ESM4adj_historical_"
    input_tair_prefix      = "./indus_data/tas_daily_GFDL-ESM4adj_historical/tas_daily_GFDL-ESM4adj_historical_"
    input_wind_prefix      = "./indus_data/wind10_daily_GFDL-ESM4_historical/wind10_daily_GFDL-ESM4_historical_"
    input_vp_prefix        = "./indus_data/vp_daily_GFDL-ESM4_historical/vp_daily_GFDL-ESM4_historical_"
    input_swdown_prefix    = "./indus_data/swdown_daily_GFDL-ESM4adj_historical/swdown_daily_GFDL-ESM4adj_historical_"
    input_lwdown_prefix    = "./indus_data/lwdown_daily_GFDL-ESM4adj_historical/lwdown_daily_GFDL-ESM4adj_historical_"
    # input_psurf_prefix     = "./indus_data/psurf_daily_GFDL-ESM4_historical/psurf_daily_GFDL-ESM4_historical_"
    
    # Input variable names (as specified in the input files' metadata)
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

    println("Running from year $start_year to year $end_year. \n")

else
    error("Unknown CASE: '$CASE'. Please provide 'global' or 'indus' (or any other case defined in init.jl) as the first argument.")
end
