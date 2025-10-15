# mGV
Modular Global VIC solver

<img width="1341" height="1244" alt="mGVcodeArchitecture" src="https://github.com/user-attachments/assets/296bd461-c28d-43f9-a887-a6c9ca94db15" />

## Install Julia on Linux

```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.5-linux-x86_64.tar.gz
tar -xvzf julia-1.10.5-linux-x86_64.tar.gz
sudo mv julia-1.10.5 /opt/julia
sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia
julia --version
```

---

## Activate the Project and Install Dependencies

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```


## Run Command

To run the program:

```bash
julia --project=. run.jl indus 1979 1982
```

[Interactive documentation AI assistant](https://chatgpt.com/g/g-67e43ed99d2881919a9ebbf2f225ee1d-mgv-assistant) (work in progress... might give wrong info :))


