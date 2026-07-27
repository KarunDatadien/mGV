const USAGE = "Usage: julia run.jl <config_file> [--nc|--netcdf|--output=nc]"

function get_output_format()
    for arg in ARGS
        if startswith(arg, "--output=")
            val = split(arg, "=")[2]
            if val in ["netcdf", "nc"]
                return :netcdf
            end
        elseif arg in ["--netcdf", "--nc"]
            return :netcdf
        end
    end
    return :zarr
end

function parse_args()
    # Notify the user if defaulting to "global"
    if length(ARGS) < 1
        error("run script requires the path to the config file as first argument.")
    end
    if length(ARGS) > 2
        error("run script requires a config file, and an optional extra argument (output format).")
    end
    config_file = ARGS[1]

    if !isabspath(config_file)
        config_file = abspath(config_file)
    end

    if !isfile(config_file)
        error("Config file '$config_file' does not exist or is not reachable from this path!")
    end

    return config_file
end

function ensure_output_directory(output_dir::String)
    # Check if the output directory exists, otherwise create it
    if !isdir(output_dir)
        println("Output directory '$output_dir' does not exist. Creating it...")
        mkpath(output_dir)
    end
end

function has_input_files(year)
    # List of input file prefixes
    input_prefixes = [
        input_prec_prefix,
        input_tair_prefix,
        input_wind_prefix,
        input_vp_prefix,
        input_swdown_prefix,
        input_lwdown_prefix
    ]

    # Check if all required files exist
    for prefix in input_prefixes
        file_path = "$(prefix)$(year).nc"
        if !isfile(file_path)
            println("WARNING: Input file for year $year not found: $file_path")
            return false
        end
    end
    return true
end

function day_to_month(day::Int, year::Int)
    # Construct the date from the year and day of the year
    date = Date(year, 1, 1) + Day(day - 1)
    return month(date)  # Extract the month from the date
end
