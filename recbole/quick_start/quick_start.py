
"""
recbole.quick_start
########################
"""
import logging
from logging import getLogger

import torch
import pickle

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color



"""
看整个程序的流程就在这里
run_recbole函数：需要输入的数据，model类型，dataset类型，超参数文件
（1）通过create_dataset函数将数据集整理为dataset，再用data_preparation将dataset整理为train，validation，test data的dataloader
（2）通过get_model和get_trainer得到训练模型所需要的model和trainer
（3）用trainer中的fit函数训练model，用trainer中的evaluate函数测试model
    （注意这里的fit是将函数直接训练epoch次，完成整个训练过程）
"""

def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    #将config内的东西传入
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    """
     以下是无关紧要的部分，一个是设置随机种子，一个是记录一下整个过程
    """
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    # dataset filtering
    """
     下面是通过输入的参数，决定构建怎么样的数据库
    """
    dataset = create_dataset(config)
    logger.info(dataset)
    """
    将数据集先划分成三份，再将三份数据集切分成多个datasampler，然后放入dataloader中，具体实现见data包
    从这里得到的train_data, valid_data, test_data是已经分好的三个data_sampler，即batch已经分好的
    """
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])

    """
    get_model做的事情就是通过输入的路径，取出需要的model的class
    将要跑的model取出,比如：kgat，sascts之类的，然后将model的类实例化成一个对象，即现在的model
    """
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)


    """
    这里各个model有自己的trainer，根据model的type的名字来选择trainer
    """
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)


    """
    将train_data和valid_data传入fit函数中进行训练，fit的具体实现见trainer中的fit函数
    训练+验证
    """
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    """
    测试结果，几乎可以不看
    """
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
    #logger适用于记录现在的结果
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }






def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result}
def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
