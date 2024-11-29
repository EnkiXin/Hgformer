import argparse
from recbole_gnn.quick_start import run_recbole_gnn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='name of models')
    #parser.add_argument('--dataset',default='ml-1m', type=str, help='name of datasets')
    parser.add_argument('--config_files',type=str, default='baseline_config_fixed/RecFormer_movie.yaml', help='config files')
    args, _ = parser.parse_known_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_gnn(model=args.model, config_file_list=[args.config_files])

