import copy
import hashlib
import logging
from pathlib import Path

import torch
from logging import getLogger
from recbole.utils import init_logger, init_seed, set_color
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer


def _dataset_interaction_fingerprint(dataset):
    """Return an order-independent fingerprint of a recommendation split."""

    user_ids = dataset.inter_feat[dataset.uid_field].detach().cpu().long()
    item_ids = dataset.inter_feat[dataset.iid_field].detach().cpu().long()
    # RecBole ids are dense and item ids are smaller than ``item_num``.  The
    # combined key therefore gives a collision-free lexicographic pair key.
    pair_keys = user_ids * int(dataset.item_num) + item_ids
    pair_keys = torch.sort(pair_keys).values.contiguous()
    digest = hashlib.sha256()
    digest.update(str(int(dataset.user_num)).encode("ascii"))
    digest.update(b":")
    digest.update(str(int(dataset.item_num)).encode("ascii"))
    digest.update(b":")
    digest.update(pair_keys.numpy().tobytes())
    return {
        "interactions": int(pair_keys.numel()),
        "sha256": digest.hexdigest(),
    }


def _split_fingerprints(train_data, valid_data, test_data):
    return {
        "train": _dataset_interaction_fingerprint(train_data.dataset),
        "valid": _dataset_interaction_fingerprint(valid_data.dataset),
        "test": _dataset_interaction_fingerprint(test_data.dataset),
    }


def run_recbole_gnn(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    evaluate_test=True,
):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset
    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        evaluate_test (bool, optional): Whether to evaluate the held-out test
            split after training. Disable this during hyperparameter search so
            trial selection is based only on validation metrics.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("Your config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    split_fingerprints = _split_fingerprints(train_data, valid_data, test_data)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )
    # Keep the held-out test split untouched while tuning. A selected
    # configuration can be rerun once with ``evaluate_test=True``.
    test_result = None
    if evaluate_test:
        test_result = trainer.evaluate(
            test_data,
            load_best_model=saved,
            show_progress=config['show_progress'],
        )

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    if evaluate_test:
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
    model_diagnostics = (
        model.projection_diagnostics()
        if callable(getattr(model, "projection_diagnostics", None))
        else None
    )
    return {
        'model': config['model'],
        'dataset': config['dataset'],
        'seed': config['seed'],
        'epochs': config['epochs'],
        'stopping_step': config['stopping_step'],
        'data_path': config['data_path'],
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result,
        'checkpoint_file': (
            str(Path(trainer.saved_model_file).resolve())
            if saved and Path(trainer.saved_model_file).is_file()
            else None
        ),
        'split_fingerprints': split_fingerprints,
        'model_diagnostics': model_diagnostics,
    }


def evaluate_recbole_gnn_checkpoint(
    checkpoint_file,
    *,
    evaluate_valid=True,
    evaluate_test=True,
    eval_batch_size=None,
    eval_user_chunk_size=None,
    eval_item_chunk_size=None,
    full_sort_user_batch_size=None,
    device=None,
    show_progress=False,
):
    """Evaluate a saved validation-selected checkpoint with full ranking.

    The checkpoint's own data filtering, random seed, ordering, grouping and
    split settings are reused.  Only the evaluation mode and memory-related
    batch/chunk settings are changed, so rebuilding the dataset yields the
    same train/validation/test edge partition used during sampled validation.
    Returned split fingerprints make that invariant auditable against the
    training result JSON.
    """

    if not evaluate_valid and not evaluate_test:
        raise ValueError("at least one of evaluate_valid/evaluate_test must be true")

    checkpoint_path = Path(checkpoint_file).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"not a RecBole training checkpoint: {checkpoint_path}")

    # torch.load creates a private Config object, but copy it anyway so this
    # helper has no observable mutation if callers retain the checkpoint dict.
    config = copy.deepcopy(checkpoint["config"])
    original_eval_args = copy.deepcopy(config["eval_args"])
    full_eval_args = copy.deepcopy(original_eval_args)
    full_eval_args["mode"] = "full"
    config["eval_args"] = full_eval_args
    config["eval_neg_sample_args"] = {
        "strategy": "full",
        "distribution": "uniform",
    }

    if eval_batch_size is not None:
        config["eval_batch_size"] = int(eval_batch_size)
    if eval_user_chunk_size is not None:
        config["eval_user_chunk_size"] = int(eval_user_chunk_size)
    if eval_item_chunk_size is not None:
        config["eval_item_chunk_size"] = int(eval_item_chunk_size)
    if full_sort_user_batch_size is not None:
        config["full_sort_user_batch_size"] = int(full_sort_user_batch_size)
    config["show_progress"] = bool(show_progress)

    if device is None:
        selected_device = torch.device(
            "cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu"
        )
    else:
        selected_device = torch.device(device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    config["device"] = selected_device
    config["use_gpu"] = selected_device.type == "cuda"

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    split_fingerprints = _split_fingerprints(train_data, valid_data, test_data)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data.dataset).to(selected_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    # Full-sort tables are deterministic caches, not learned parameters.  Old
    # checkpoints may contain them on a different device or from a different
    # evaluation batch, so always rebuild them lazily.
    if hasattr(model, "_clear_full_sort_cache"):
        model._clear_full_sort_cache()
    logger.info(model)

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.data_collect(train_data)
    valid_result = None
    test_result = None
    if evaluate_valid:
        valid_result = trainer.evaluate(
            valid_data,
            load_best_model=False,
            show_progress=show_progress,
        )
    if evaluate_test:
        test_result = trainer.evaluate(
            test_data,
            load_best_model=False,
            show_progress=show_progress,
        )

    logger.info(set_color("full-ranking valid result", "yellow") + f": {valid_result}")
    logger.info(set_color("full-ranking test result", "yellow") + f": {test_result}")
    model_diagnostics = (
        model.projection_diagnostics()
        if callable(getattr(model, "projection_diagnostics", None))
        else None
    )
    return {
        "model": config["model"],
        "dataset": config["dataset"],
        "seed": config["seed"],
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_valid_score": checkpoint.get("best_valid_score"),
        "selection_eval_mode": original_eval_args.get("mode"),
        "evaluation_eval_mode": "full",
        "eval_batch_size": config["eval_batch_size"],
        "eval_user_chunk_size": config["eval_user_chunk_size"],
        "eval_item_chunk_size": config["eval_item_chunk_size"],
        "full_sort_user_batch_size": (
            config["full_sort_user_batch_size"]
            if "full_sort_user_batch_size" in config
            else None
        ),
        "split_fingerprints": split_fingerprints,
        "valid_result": valid_result,
        "test_result": test_result,
        "model_diagnostics": model_diagnostics,
    }
def objective_function(
    config_dict=None,
    config_file_list=None,
    saved=True,
    evaluate_test=True,
):
    r""" The default objective_function used in HyperTuning
    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        evaluate_test (bool, optional): Whether to evaluate the held-out test
            split after fitting. Hyperparameter selection must set this to
            ``False`` so that every trial is validation-only. Defaults to
            ``True`` for backward compatibility with the original runner.
    """
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("Your config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = None
    if evaluate_test:
        test_result = trainer.evaluate(test_data, load_best_model=saved)
    return {
        'model': config['model'],
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
