# mGV intro
mGV (modular Global VIC solver) is a high-performance global macroscale hydrologic model programmed in Julia, based on the VIC framework by Liang et al. (1994). The code in this repo is a work-in-progress and in it's early development phase. It includes GPU acceleration, currently supporting NVIDIA/CUDA only.

# Code structure overview
<img width="1341" height="1244" alt="mGVcodeArchitecture" src="https://github.com/user-attachments/assets/296bd461-c28d-43f9-a887-a6c9ca94db15" />


## How to run mGV

### Install Julia on Linux (if not available yet on your machine)

```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.5-linux-x86_64.tar.gz
tar -xvzf julia-1.10.5-linux-x86_64.tar.gz
sudo mv julia-1.10.5 /opt/julia
sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia
julia --version
```

---

### Activate the Project and Install Dependencies
Needs to be done only once.

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```


### Run Command

To run the program:

```bash
julia --project=. run.jl mekong 1979 1980
```

The program currently requires an NVIDIA GPU, if it doesn't detect one, it will automatically abort the run. This will provide output netCDF files for the Mekong region, years 1979 and 1980. Currently we only provide forcing and landsurface parameter files for the (small) Mekong region in this repo, due to file-size considerations. Data for the entire globe and the indus region will be made available at a later point in development.
