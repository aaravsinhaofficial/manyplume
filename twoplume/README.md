Command to generate training data for 2plume:
python sim_cli.py --duration 100 --cores 1 --dataset_name twoplume --fname_suffix "" --dt 0.01 --wind_magnitude 0.1 --wind_y_varx 1.0 --birth_rate 0.2 --outdir "$(python -c 'import config; print(config.datadir)')" --source_positions "-1,-1;-1,1" --warmup_time 100.0
