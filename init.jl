using NCDatasets
using ProgressMeter
using CUDA
using Dates  # For timestamping logs

#include("physics.jl")
#using .PhysicsFunctions  # For GPU-accelerated functions

# Define file paths
input_precip_prefix = "./global_data/5arcmin/forcing/prec/prec_WFDE5_CRU+GPCC_v2.0_5arcmin_"
input_tair_prefix = "./global_data/5arcmin/forcing/tair/tair_WFDE5_v2.0_5arcmin_"
input_param_prefix = "./global_data/vic_global_5min_params_fix2.nc"
output_dir = "./output/"
start_year = 1979
end_year = 2019