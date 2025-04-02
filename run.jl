# Load user-modifiable configuration
include("config/user_config.jl")

# Load package definitions
include("src/packages.jl")

# Constants
include("src/constants/sim_constants.jl")
include("src/constants/physical_constants.jl")

# Utilities
include("src/utils/runtime_utils.jl")

# I/O Helpers
include("src/io/io_helpers.jl")
include("src/io/netcdf_writer.jl")
include("src/io/parameter_reader.jl")

# Initialization routines
include("src/init/init_calc.jl")

# Physics Modules
include("src/physics/physics.jl")
include("src/physics/evapotranspiration.jl")
include("src/physics/groundwater.jl")
include("src/physics/temperature.jl")

# Main program entry point
include("src/main.jl")