import argparse
import shlex
import sys
from recbole.quick_start import run_recbole

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='name of model')
    parser.add_argument('-d', '--dataset', type=str, help='name of dataset')
    parser.add_argument(
        '--config-files', '--config_files',
        dest='config_files',
        type=str,
        default='baseline_config_fixed/SimpleX.yaml',
        help='one or more config files (quote a space-separated list)',
    )
    parser.add_argument('--no-save', action='store_true', help='do not save a checkpoint')
    args, recbole_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *recbole_args]
    config_file_list = shlex.split(args.config_files) if args.config_files else None
    run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        saved=not args.no_save,
    )
