from contextlib import contextmanager
import random
from time import time
import math

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from recbole.trainer import Trainer
from recbole.utils import early_stopping, dict2str, set_color, get_gpu_usage


@contextmanager
def _temporary_validation_rng(seed):
    """Use a deterministic RNG stream without advancing the training stream.

    Legacy RecBole negative samplers draw from the process-wide NumPy RNG.
    Some model/evaluation code may also use Python or Torch randomness, so all
    initialized streams are isolated together.  CUDA is deliberately left
    untouched when it has not been initialized yet.
    """

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_was_initialized = torch.cuda.is_available() and torch.cuda.is_initialized()
    cuda_states = torch.cuda.get_rng_state_all() if cuda_was_initialized else None

    # NumPy's legacy seed accepts uint32 values.  Use a private CPU Generator
    # to avoid torch.manual_seed's side effect on not-yet-initialized CUDA RNGs.
    numpy_seed = int(seed) % (2 ** 32)
    torch_seed = int(seed) % (2 ** 63)
    seeded_torch_state = torch.Generator(device="cpu").manual_seed(torch_seed).get_state()
    try:
        np.random.seed(numpy_seed)
        random.seed(int(seed))
        torch.random.set_rng_state(seeded_torch_state)
        if cuda_was_initialized:
            torch.cuda.manual_seed_all(torch_seed)
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class SLRecGraphTrainer(Trainer):
    """Trainer with opt-in fixed candidates for sampled validation only.

    RecBole's ``RepeatableSampler`` is not a cached evaluation sampler: it
    still draws fresh values from ``numpy.random`` on every loader pass.  For
    early stopping that makes successive validation scores incomparable.  The
    opt-in below replays one validation-only RNG stream while restoring the
    training RNG streams exactly afterwards.
    """

    def _fixed_sampled_validation_enabled(self):
        if not bool(self.config["fixed_sampled_validation"]):
            return False

        eval_args = self.config["eval_args"] or {}
        eval_mode = eval_args.get("mode")
        eval_neg_sample_args = self.config["eval_neg_sample_args"] or {}
        return (
            isinstance(eval_mode, str)
            and "full" not in eval_mode.lower()
            and eval_neg_sample_args.get("strategy") == "by"
        )

    def _fixed_sampled_validation_seed(self):
        seed = self.config["fixed_sampled_validation_seed"]
        if seed is None:
            seed = self.config["seed"]
        if seed is None:
            seed = 0
        try:
            return int(seed)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "fixed_sampled_validation_seed must be an integer"
            ) from error

    def _valid_epoch(self, valid_data, show_progress=False):
        if not self._fixed_sampled_validation_enabled():
            return super()._valid_epoch(valid_data, show_progress=show_progress)

        with _temporary_validation_rng(self._fixed_sampled_validation_seed()):
            return super()._valid_epoch(valid_data, show_progress=show_progress)


class NCLTrainer(Trainer):
    def __init__(self, config, model):
        super(NCLTrainer, self).__init__(config, model)

        self.num_m_step = config['m_step']
        assert self.num_m_step is not None

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.
        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.
        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)

        for epoch_idx in range(self.start_epoch, self.epochs):

            # only differences from the original trainer
            if epoch_idx % self.num_m_step == 0:
                self.logger.info("Running E-step ! ")
                self.model.e_step()
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch
        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.
        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                if epoch_idx < self.config['warm_up_step']:
                    losses = losses[:-1]
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss


class HMLETTrainer(Trainer):
    def __init__(self, config, model):
        super(HMLETTrainer, self).__init__(config, model)

        self.warm_up_epochs = config['warm_up_epochs']
        self.ori_temp = config['ori_temp']
        self.min_temp = config['min_temp']
        self.gum_temp_decay = config['gum_temp_decay']
        self.epoch_temp_decay = config['epoch_temp_decay']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if epoch_idx > self.warm_up_epochs:
            # Temp decay
            gum_temp = self.ori_temp * math.exp(-self.gum_temp_decay*(epoch_idx - self.warm_up_epochs))
            self.model.gum_temp = max(gum_temp, self.min_temp)
            self.logger.info(f'Current gumbel softmax temperature: {self.model.gum_temp}')

            for gating in self.model.gating_nets:
                self.model._gating_freeze(gating, True)
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)


class SEPTTrainer(Trainer):
    def __init__(self, config, model):
        super(SEPTTrainer, self).__init__(config, model)
        self.warm_up_epochs = config['warm_up_epochs']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if epoch_idx < self.warm_up_epochs:
            loss_func = self.model.calculate_rec_loss
        else:
            self.model.subgraph_construction()
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)
