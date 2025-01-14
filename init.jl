# Use the first argument in ARGS as the CASE, defaulting to "global" if not provided
const CASE = length(ARGS) > 0 ? ARGS[1] : "global"

# Notify the user if defaulting to "global"
if length(ARGS) == 0
    println("No CASE provided. Defaulting to 'global'.")
end

# Parse optional start and end year from ARGS
const start_year_arg = length(ARGS) > 1 ? parse(Int, ARGS[2]) : nothing
const end_year_arg   = length(ARGS) > 2 ? parse(Int, ARGS[3]) : nothing

# Load the configuration matching the input argument:
if CASE == "global"

    println("Loading configuration for 'global'...")

    # =========================== GLOBAL CONFIGURATION ===========================
        
    # Input file paths/names
    input_param_file       = "./global_data/vic_global_5min_params_fix2.nc"
    input_precip_prefix    = "./global_data/5arcmin/forcing/prec/prec_WFDE5_CRU+GPCC_v2.0_5arcmin_"
    input_tair_prefix      = "./global_data/5arcmin/forcing/tair/tair_WFDE5_v2.0_5arcmin_"
    input_wind_prefix      = "./global_data/5arcmin/forcing/wind/wind_WFDE5_v2.0_5arcmin_"
    input_vp_prefix        = "./global_data/5arcmin/forcing/vp/vp_WFDE5_v2.0_5arcmin_"
    input_swdown_prefix    = "./global_data/5arcmin/forcing/swdown/swdown_WFDE5_v2.0_5arcmin_"
    input_lwdown_prefix    = "./global_data/5arcmin/forcing/lwdown/lwdown_WFDE5_v2.0_5arcmin_"
    
    # Output file paths/names
    output_dir             = "./output/"
    output_file_prefix     = "outputfile_global_"
    
    # Set default simulation years if no command-line arguments are provided
    start_year             = isnothing(start_year_arg) ? 1979 : start_year_arg
    end_year               = isnothing(end_year_arg)   ? 2019 : end_year_arg
    
    # ========================= END GLOBAL CONFIGURATION ==========================

    println("Running from year $start_year to year $end_year.")

elseif CASE == "indus"

    println("Loading configuration for 'indus'...")
    
    # ============================ INDUS CONFIGURATION ============================
   
    # Input file paths/names
    input_param_file       = "./indus_data/domain_Indus.nc"
    input_precip_prefix    = "./indus_data/pr_daily_GFDL-ESM4adj_historical/pr_daily_GFDL-ESM4adj_historical_"
    input_tair_prefix      = "./indus_data/tas_daily_GFDL-ESM4adj_historical/tas_daily_GFDL-ESM4adj_historical_"
    input_wind_prefix      = "./indus_data/wind10_daily_GFDL-ESM4_historical/wind10_daily_GFDL-ESM4_historical_"
    input_vp_prefix        = "./indus_data/vp_daily_GFDL-ESM4_historical/vp_daily_GFDL-ESM4_historical_"
    input_swdown_prefix    = "./indus_data/swdown_daily_GFDL-ESM4adj_historical/swdown_daily_GFDL-ESM4adj_historical_"
    input_lwdown_prefix    = "./indus_data/lwdown_daily_GFDL-ESM4adj_historical/lwdown_daily_GFDL-ESM4adj_historical_"
    input_psurf_prefix     = "./indus_data/psurf_daily_GFDL-ESM4_historical/psurf_daily_GFDL-ESM4_historical_"
    
    # Output file paths/names
    output_dir             = "./output_indus/"
    output_file_prefix     = "outputfile_indus_"
    
    # Set default simulation years if no command-line arguments are provided
    start_year             = isnothing(start_year_arg) ? 1979 : start_year_arg
    end_year               = isnothing(end_year_arg)   ? 2010 : end_year_arg
    
    # ========================== END INDUS CONFIGURATION ==========================

    println("Running from year $start_year to year $end_year.")

else
    error("Unknown CASE: '$CASE'. Please provide 'global' or 'indus' (or any other case defined in init.jl) as the first argument.")
end
