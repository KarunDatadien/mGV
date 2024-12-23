# Asynchronously compresses a NetCDF file using `nccopy`, offloading compression to the shell for efficiency.
# Accepts `output_file` and `compression_level` (1 = fast/light, 9 = slow/max).
function compress_file_async(output_file::String, compression_level::Int)
    println("Attempting compression of file $output_file asynchronously with compression level $compression_level...")

    # Construct the shell command as a string
    command = """
    nohup bash -c 'nccopy -d $compression_level $output_file $output_file.tmp && mv $output_file.tmp $output_file' > /dev/null 2>&1 &
    """

    # Run the detached shell command
    try
        run(`sh -c $command`)
        println("Compression process for $output_file started in the background.")
    catch err
        println("Failed to start compression for $output_file: $err")
    end
end
