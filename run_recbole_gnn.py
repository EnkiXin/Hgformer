import argparse
import json
import shlex
import sys
from pathlib import Path
from recbole_gnn.quick_start import run_recbole_gnn


def _json_default(value):
    if hasattr(value, 'item'):
        return value.item()
    return str(value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='name of model')
    parser.add_argument('-d', '--dataset', type=str, help='name of dataset')
    parser.add_argument(
        '--config-files', '--config_files',
        dest='config_files',
        type=str,
        default='baseline_config_fixed/RecFormer_movie.yaml',
        help='one or more config files (quote a space-separated list)',
    )
    parser.add_argument('--no-save', action='store_true', help='do not save a checkpoint')
    parser.add_argument(
        '--validation-only',
        action='store_true',
        help='skip held-out test evaluation (use this for hyperparameter search)',
    )
    parser.add_argument(
        '--result-file',
        type=str,
        help='also write the returned validation/test metrics as JSON',
    )
    args, recbole_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *recbole_args]
    config_file_list = shlex.split(args.config_files) if args.config_files else None
    result = run_recbole_gnn(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        saved=not args.no_save,
        evaluate_test=not args.validation_only,
    )
    if args.result_file:
        output = Path(args.result_file).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'model': args.model,
            'dataset': args.dataset,
            'config_files': config_file_list,
            **result,
        }
        output.write_text(
            json.dumps(payload, indent=2, default=_json_default) + '\n',
            encoding='utf-8',
        )
        print(f'RESULT_JSON={output.resolve()}')
