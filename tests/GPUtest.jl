using NCDatasets
using ProgressMeter
using CUDA
using Dates  # For timestamping logs

# Initialize GPU to ensure CUDA is properly set up
CUDA.zeros(Float32, 1)  # Allocate a small array to initialize GPU memory manager

# Define file paths
input_precip_dir = "./global_data/5arcmin/forcing/prec/"
input_param_file = "./global_data/vic_global_5min_params_fix2.nc"
output_dir = "./output/"
start_year = 1979
end_year = 2019

# Loop over years and process precipitation data
for year in start_year:end_year
    precip_file = joinpath(input_precip_dir, "prec_WFDE5_CRU+GPCC_v2.0_5arcmin_$(year).nc")
    output_file = joinpath(output_dir, "precipitation_scaled_$(year).nc")

    println("Processing year: $year (GPU Optimized)")
    println("Input file: $precip_file")
    println("Output file: $output_file")

    # Open the input NetCDF file
    NCDataset(precip_file) do ds
        # Read the precipitation variable (assuming it's named "pr")
        precipitation = ds["prec"]

        # Get dimensions (time only; ignore original lon and lat)
        num_days = size(precipitation, 3)
        println("Original shape of input precipitation: ", size(precipitation))

        # Create a new NetCDF file to store the scaled data
        NCDataset(output_file, "c") do out_ds

            # Define dimensions and variables
            defDim(out_ds, "lon", size(precipitation, 1))
            defDim(out_ds, "lat", size(precipitation, 2))
            defDim(out_ds, "time", size(precipitation, 3))
            defDim(out_ds, "layer", 3)

            # Define a new variable for scaled precipitation
            pr_scaled = defVar(out_ds, "scaled_precipitation", Float32, ("lon", "lat", "time"),
                               deflatelevel=1, chunksizes=(512, 512, 1), fillvalue=-9999.0f0)

            println("Writing scaled precipitation data slice-by-slice to NetCDF for year $year...")

            # Allocate GPU arrays only once outside the loop
            d_data = CUDA.zeros(Float32, size(precipitation, 1), size(precipitation, 2))

            # Step 1: Transfer the precipitation slice for the current day to the GPU
            @time cpu_data_cleaned = Float32.(replace(precipitation[:, :, :], missing => NaN))

            # d_precipitation = CUDA.CuArray(cpu_data_cleaned)  # Transfer the entire dataset to the GPU

            # Process and write data one day at a time
            @showprogress "Processing year $year (GPU)..." for day in 1:num_days

                # Step 2: Explicitly copy the cleaned data to the GPU
                CUDA.copyto!(d_data, cpu_data_cleaned[:,:,day]) #Time: 0:01:51, takes hardly any GPU memory: total 537MiB (and baseline, doing nothign: 455MiB) <- 100 loops | 500 loops -> Time: 0:03:45
                # d_data .= view(d_precipitation, :, :, day) #Time: 0:01:34, takes significant GPU memory: 10567MiB

                # Step 2: Perform heavy transformations ONCE on the GPU
                for _ in 1:500
                    d_data .= sqrt.(max.(d_data, 0) .+ 1)       # Square root transformation
                    d_data .= log.(d_data .+ 1e-6)              # Logarithmic transformation
                    d_data .= exp.(d_data) .* sin.(d_data)      # Non-linear exp and sine combination
                    d_data .= sqrt.(max.(d_data, 0) .+ 1)       # Square root transformation
                    d_data .= log.(d_data .+ 1e-6)              # Logarithmic transformation
                    d_data .= exp.(d_data) .* sin.(d_data)      # Non-linear exp and sine combination
                    d_data .= sqrt.(max.(d_data, 0) .+ 1)       # Square root transformation
                    d_data .= log.(d_data .+ 1e-6)              # Logarithmic transformation
                    d_data .= exp.(d_data) .* sin.(d_data)      # Non-linear exp and sine combination
                    d_data .= sqrt.(max.(d_data, 0) .+ 1)       # Square root transformation
                    d_data .= log.(d_data .+ 1e-6)              # Logarithmic transformation
                    d_data .= exp.(d_data) .* sin.(d_data)      # Non-linear exp and sine combination
                end

                # Step 3: Write results directly to the NetCDF file from the GPU
                pr_scaled[:, :, day] = Array(d_data)  # Transfer only final results back to CPU
            end

            println("Data successfully written for year: $year.")
            # Add attributes to the variable
            pr_scaled.attrib["units"] = "mm/day"
            pr_scaled.attrib["description"] = "Daily precipitation scaled with GPU computations (optimized)"
        end
    end

    println("Completed processing for year: $year\n")
end

println("All files processed successfully with GPU optimization.")
