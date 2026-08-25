set -e

julia --project=. -e 'using mGV; mGV.run()' -- configs/mekong_config.toml
#julia --project=. -e 'using mGV; mGV.run()' -- configs/indus_config.toml --format=$FORMAT

python3 validations/plot_dashboard.py mekong
