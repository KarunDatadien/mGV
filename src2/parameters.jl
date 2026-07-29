struct GridParameters{V <: AbstractVector, M <: AbstractMatrix, T <: AbstractArray}
    # Static
    latitude::V
    longitude::V
    elevation::M
    average_temperature::M
    annual_precipitation::M
    snow_band_area_fraction::T
    snow_band_elevation::T
    snow_band_precipitation_factor::T
end

@adapt_structure GridParameters

struct VegetationParameters{T <: AbstractArray}
    # Static
    root_fraction::T
    vegetation_fraction::T
    minimum_resistance::T
    architectural_resistance::T
    # Active monthly values
    displacement_height::T
    roughness_length::T
    lai::T
    albedo::T
    canopy_coverage::T
end

@adapt_structure VegetationParameters

struct SoilParameters{M <: AbstractMatrix, T <: AbstractArray}
    # Static soil and baseflow parameters
    hydraulic_conductivity::T
    depth::T
    initial_moisture::T
    residual_moisture::T
    critical_moisture_fraction::T
    field_capacity_fraction::T
    wilting_point_fraction::T
    quartz_content::T
    bare_roughness::M
    bulk_density::T
    particle_density::T
    campbell_n::T
    nijssen_infilt_b::M
    nijssen_lin_reservoir::M
    nijssen_nolin_reservoir::M
    moisture_depth_baseflow_transition::M
    column_depth::M
    baseflow_curve_exp::M
end

@adapt_structure SoilParameters

function read_parameters(config::Cfg)
    ds_params = NCDataset(config.input.paths.input_param_file)
    grid_params = GridParameters(
        nomissing(ds_params[cfg.input.names.latitude][:], 0.0),
        nomissing(ds_params[cfg.input.names.longitude][:], 0.0),
        nomissing(ds_params[cfg.input.names.elevation][:,:], 0.0),
        nomissing(ds_params[cfg.input.names.average_temperature][:,:], 0.0),
        nomissing(ds_params[cfg.input.names.annual_precipitation][:,:], 0.0),
        nomissing(ds_params[cfg.input.names.snow_band_area_fraction][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.snow_band_elevation][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.snow_band_precipitation_factor][:,:,:], 0.0),
    )

    # Reshape some 3D inputs to 4D
    cv = nomissing(ds_params[cfg.input.names.vegetation_fraction][:,:,:], 0.0)
    vegetation_fraction = ndims(cv) == 3 ? reshape(cv, size(cv, 1), size(cv, 2), 1, size(cv, 3)) : cv
    rmin = nomissing(ds_params[cfg.input.names.minimum_resistance][:,:,:], 0.0)
    minimum_resistance = ndims(rmin) == 3 ? reshape(rmin, size(rmin, 1), size(rmin, 2), 1, size(rmin, 3)) : rmin
    rarc = nomissing(ds_params[cfg.input.names.architectural_resistance][:,:,:], 0.0)
    architectural_resistance = ndims(rarc) == 3 ? reshape(rarc, size(rarc, 1), size(rarc, 2), 1, size(rarc, 3)) : rarc

    veg_params = VegetationParameters(
        nomissing(ds_params[cfg.input.names.root_fraction][:,:,:,:], 0.0),
        vegetation_fraction,
        minimum_resistance,
        architectural_resistance,
        nomissing(ds_params[cfg.input.names.displacement_height][:,:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.roughness_length][:,:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.lai][:,:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.albedo][:,:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.canopy_coverage][:,:,:,:], 0.0)
    )

    soil_params = SoilParameters(
        nomissing(ds_params[cfg.input.names.hydraulic_conductivity][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.depth][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.initial_moisture][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.residual_moisture][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.critical_moisture_fraction][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.field_capacity_fraction][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.wilting_point_fraction][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.quartz_content][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.bare_roughness][:,:], 0.0),
        nomissing(ds_params[cfg.input.names.bulk_density][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.particle_density][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.campbell_n][:,:,:], 0.0),
        nomissing(ds_params[cfg.input.names.nijssen_infilt_b][:,:], 0.0),
        nomissing(ds_params[cfg.input.names.nijssen_lin_reservoir][:,:], 0.0),
        nomissing(ds_params[cfg.input.names.nijssen_nolin_reservoir][:,:], 0.0),
        nomissing(ds_params[cfg.input.names.moisture_depth_baseflow_transition][:,:], 0.0),
        nomissing(ds_params[cfg.input.names.column_depth][:,:], 0.0),
        nomissing(ds_params[cfg.input.names.baseflow_curve_exp][:,:], 0.0)
    )
    return grid_params, veg_params, soil_params
end

function read_and_allocate_parameter(varname::String)
    println("Loading $varname parameter input...")

    # 1) Open netCDF file 
    dataset = NetCDF.open(input_param_file)
    
    if !haskey(dataset.vars, varname)
        println("  -> WARNING: Variable $varname not found in $input_param_file. Returning nothing.")
        return nothing, nothing
    end

    var_dims = size(dataset[varname])

    # 2) Read data sequentially into memory and format
    slicing_indices = repeat([:], length(var_dims))
    cpu_preload = dataset[varname][slicing_indices...]

    # NetCDF.jl natively replaces _FillValue with Float NaN. This causes math corruption cascades. Re-cast to 0.0.
    cpu_preload[isnan.(cpu_preload)] .= eltype(cpu_preload)(0.0)

    # 3) Print array sizes for diagnostics
    if length(var_dims) <= 4
        println("Element type for $(length(var_dims))D: ", eltype(cpu_preload))
    end
    println("Full size of $varname: ", size(cpu_preload))

    # 4) Optimizations for data transfer
    # Locks memory pages if using NVIDIA/AMD; does nothing on CPU/Metal.
    try
        pin_memory!(cpu_preload)
    catch e
        println("  -> WARNING: Failed to pin CPU memory. Transfer will be slower. Error: $e")
    end

    # 5) Allocate device memory
    # Handle 4D reshaping logic to only pre-allocate memory daily for monthly (vegetation) tiles 
    adjusted_dims = if length(var_dims) == 4
        (var_dims[1], var_dims[2], (var_dims[3] == 12 ? 1 : var_dims[3]), var_dims[4])
    else
        var_dims
    end

    # 6) Pre-allocating memory on the active device (VRAM/RAM)
    device_arr = alloc(FloatType, adjusted_dims...)
    println("Allocated $backend_name array of size: ", size(device_arr))

    return cpu_preload, device_arr
end
