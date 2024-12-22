# Asynchronously compresses a NetCDF file using `nccopy`, offloading compression to the shell for efficiency.
# Accepts `output_file` and `compression_level` (1 = fast/light, 9 = slow/max).
function compress_file_async(output_file::String, compression_level::Int)
    @async begin
        try
            println("Compressing file $output_file asynchronously with compression level $compression_level...")
            @time run(`nccopy -d $compression_level $output_file $output_file.tmp`)
            mv("$output_file.tmp", "$output_file") 
            println("Compression of $output_file completed successfully!")
        catch err
            println("Compression failed for $output_file: $err")
        end
    end
end