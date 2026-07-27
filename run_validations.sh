julia --project=. run.jl configs/mekong_config.toml --nc
#julia --project=. run.jl configs/indus_config.toml --nc

python3 validations/plot_dashboard.py mekong
