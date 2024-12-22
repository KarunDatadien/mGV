# Helper function to read a NetCDF variable
function read_netcdf_variable(prefix, year, variable_name)
    file_path = "$(prefix)$(year).nc"
    dataset = NCDataset(file_path, "r")
    return dataset, dataset[variable_name]  # Return both the dataset and the variable
end