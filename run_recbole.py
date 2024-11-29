import argparse
from recbole.quick_start import run_recbole

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files',type=str, default='baseline_config_fixed/SimpleX.yaml', help='config files')
    args, _ = parser.parse_known_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(config_file_list=[args.config_files])