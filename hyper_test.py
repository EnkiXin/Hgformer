import argparse
import logging
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
import warnings
warnings.filterwarnings('ignore')
from recbole.trainer import HyperTuning
from recbole_gnn.quick_start import objective_function


init_seed(2024,True)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fixed',default='baseline_config_fixed/HCF_cd.yaml', type=str, help='fixed config files')
    parser.add_argument('--config_flexible',default='baseline_config_flexible/HCF.test', type=str,  help='flexible config files')
    parser.add_argument('--output_file',default='baseline_results/hcf_HGCFAmazonCD', type=str, help='output file')
    args, _ = parser.parse_known_args()
    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    parameter_dict = {
                      }
    hp = HyperTuning(objective_function=objective_function,
                     algo='exhaustive',
                     max_evals=100,
                     params_file=args.config_flexible,
                     params_dict=parameter_dict,
                     fixed_config_file_list=[args.config_fixed])
    hp.run()
    hp.export_result(output_file=args.output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])
if __name__ == '__main__':
    main()