julia --project=. run.jl configs/mekong_config.toml 1979 1980 --nc
#julia --project=. run.jl configs/indus_config.toml --nc

cd ./validations
#python3 plot_dashboards_mekong_indus.py
python3 plot_dashboards_mekong.py
