julia --project=. run.jl configs/mekong_config.toml 1979 1980 --nc
#julia --project=. run.jl configs/indus_config.toml --nc

python3 validations/plot_dashboard.py mekong
