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
            println("⚠️ WARNING: Input file for year $year not found: $file_path")
            return false
        end
    end
    return true
end

function reshape_static_inputs!()
    global rmin_gpu, rarc_gpu

    # Check and reshape rmin_gpu if it has 3 dimensions.
    if ndims(rmin_gpu) == 3
        rmin_gpu = CUDA.reshape(rmin_gpu,
                                size(rmin_gpu, 1),
                                size(rmin_gpu, 2),
                                1,
                                size(rmin_gpu, 3))
        println("rmin_gpu reshaped to: ", size(rmin_gpu))
    else
        println("rmin_gpu already has ", ndims(rmin_gpu), " dimensions; no reshape needed.")
    end

    # Check and reshape rarc_gpu if it has 3 dimensions.
    if ndims(rarc_gpu) == 3
        rarc_gpu = CUDA.reshape(rarc_gpu,
                                size(rarc_gpu, 1),
                                size(rarc_gpu, 2),
                                1,
                                size(rarc_gpu, 3))
        println("rarc_gpu reshaped to: ", size(rarc_gpu))
    else
        println("rarc_gpu already has ", ndims(rarc_gpu), " dimensions; no reshape needed.")
    end
end