import argparse
from functools import partial
import json
import logging
import os
from pathlib import Path
import shlex
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
import warnings
warnings.filterwarnings('ignore')
from recbole.trainer import HyperTuning
from recbole_gnn.quick_start import objective_function


init_seed(2024,True)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    raise TypeError(f"cannot persist tuning value of type {type(value)!r}")


class CachedObjective:
    """Persist each completed native HyperTuning trial atomically."""

    def __init__(self, objective, cache_path):
        self.objective = objective
        self.cache_path = Path(cache_path).expanduser().resolve()
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}

    @staticmethod
    def key(config_dict):
        return json.dumps(_json_safe(config_dict), sort_keys=True, separators=(",", ":"))

    def __call__(self, config_dict=None, config_file_list=None):
        trial_key = self.key(config_dict or {})
        cached = self.cache.get(trial_key)
        if cached is not None:
            print(f"reusing completed trial: {trial_key}")
            return cached
        result = _json_safe(self.objective(config_dict, config_file_list))
        self.cache[trial_key] = result
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_name(self.cache_path.name + ".tmp")
        temporary.write_text(
            json.dumps(self.cache, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.cache_path)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fixed',default='baseline_config_fixed/HCF_cd.yaml', type=str, help='one or more fixed config files (quote a space-separated list)')
    parser.add_argument('--config_flexible',default='baseline_config_flexible/HCF.test', type=str,  help='flexible config files')
    parser.add_argument('--output_file',default='baseline_results/hcf_HGCFAmazonCD', type=str, help='output file')
    parser.add_argument(
        '--validation-only',
        action='store_true',
        help='never evaluate the held-out test split in tuning trials',
    )
    parser.add_argument(
        '--trial-cache',
        type=str,
        default=None,
        help='JSON file used to persist and reuse every completed trial',
    )
    args, _ = parser.parse_known_args()
    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    parameter_dict = {
                      }
    fixed_config_file_list = shlex.split(args.config_fixed)
    if not fixed_config_file_list:
        parser.error('--config_fixed must contain at least one config path')
    tuning_objective = partial(
        objective_function,
        evaluate_test=not args.validation_only,
    )
    if args.trial_cache:
        tuning_objective = CachedObjective(tuning_objective, args.trial_cache)
    hp = HyperTuning(objective_function=tuning_objective,
                     algo='exhaustive',
                     max_evals=100,
                     params_file=args.config_flexible,
                     params_dict=parameter_dict,
                     fixed_config_file_list=fixed_config_file_list)
    hp.run()
    Path(args.output_file).expanduser().resolve().parent.mkdir(
        parents=True, exist_ok=True
    )
    hp.export_result(output_file=args.output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])
if __name__ == '__main__':
    main()
