set -e

julia --project=. -e 'using mGV; mGV.run()' configs/mekong_config.toml --nc
#julia --project=. -e 'using mGV; mGV.run()' configs/indus_config.toml --nc

python3 validations/plot_dashboard.py mekong
