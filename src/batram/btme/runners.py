import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from batram.base_functions import TransportMap
from batram.datautils import Dataset, MinibatchSample
from batram.stopper import EarlyStopper

# the following two lines refer to Dan's private version of batram,
# file:///home/danjd/projects/transport_maps/batram-cov
# I'm commenting them out.
# from batram.cholesky import CholeskyDecompositionError
# from batram.covariate_tm import CovariateTM


class CholeskyDecompositionError(Exception):
    """Replacement for the out-commented import form Dan's private code."""

    pass


class LRScheduler(Protocol):
    def step(self) -> None:
        ...


class Runner(ABC):
    """Abstract base class for model runners."""

    model: TransportMap
    save_path: Path | str | None
    save_every: int | None
    data_size: int

    def save_model(self, epoch: int, model_name: str | None = None) -> None:
        """Helper function to save a model every save_every epochs when specified.

        If save_path or save_every is None, no checkpoints are saved.
        """
        if model_name is None:
            model_name = type(self.model).__name__

        if self.save_path is None or self.save_every is None:
            return None
        if epoch % self.save_every == 0:
            target = os.path.join(self.save_path, f"{model_name}_epoch{epoch}.pt")
            torch.save(self.model.state_dict(), target)
            return None

    def get_batch_index(
        self, batch_size: int, device: torch.device
    ) -> Sequence[torch.Tensor]:
        """Return a collated list of minibatch indices based on batch size.

        Always uses the last minibatch regardless of whether it is full or not.
        """
        random_indices = torch.randperm(self.data_size, device=device)
        batch_indices = torch.split(random_indices, batch_size)
        return batch_indices

    @abstractmethod
    def score_train_data(self, minibatch: MinibatchSample) -> float:
        """Scores model on an unweighted minibatch of training data."""
        ...

    @abstractmethod
    def score_test_data(self, test_data: Dataset) -> float:
        """Scores test data using negative log likelihood averaged over samples."""
        ...

    @abstractmethod
    def fit_model(
        self,
        train_data: Dataset,
        test_data: Dataset,
        num_epochs: int,
        batch_size: int | None = None,
        model_name: str | None = None,
    ) -> np.ndarray:
        """Fits a model to data using test_data for validation.

        Args:
        -----
        train_data: Training dataset to run the model on (should be the same as
            the model's dataset).
        test_data: Test dataset to use for validation
        """
        ...


class ModelRunner(Runner):
    """Model runner for transport map models.

    This is a generic runner for any transport map, and can be used as a first
    try before using other runners for training.

    Attributes:
        model (TransportMap): The model to fit.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler | None): The learning rate
            scheduler to use for training.
        stopper (EarlyStopper | None): The early stopping criterion to use for training.
        save_path (Path | str | None): The path to save checkpoints to.
        save_every (int | None): The number of epochs between checkpoints.
    """

    def __init__(
        self,
        model: TransportMap,
        init_lr: float = 1e-2,
        stopper_patience: int = 1000,
        save_path: Path | str | None = None,
        save_every: int | None = None,
        **kwargs,
    ):
        """Instantiates the runner class to fit a transport map.

        Arguments:
        ----------
        model: TransportMap
            Any transport map in the batram package.

        init_lr: float
            The initial learning rate to initialize AdamW with.

        stopper_patience: int
            The number of epochs to wait before stopping the optimization.

        save_path: Path | str | None
            The path to save checkpoints to.

        save_every: int | None
            The number of epochs between checkpoints.

        kwargs:
        -------
        keyword arguments to pass to the optimizer and learning rate scheduler.
        The optimizer is AdamW and the learning rate scheduler is
        CosineAnnealingLR from pytorch.

        AdamW args:
        betas: tuple (default (0.9, 0.999))
            The betas to use for the AdamW optimizer.

        eps: float (default 1e-8)
            The epsilon to use for the AdamW optimizer.

        weight_decay: float (default 0.0)
            The weight decay to use for the AdamW optimizer.


        CosineAnnealingLR args.
        T_max: int (default 1000)
            The number of steps in the learning rate schedule.

        eta_min: float (default 3e-6)
            The minimum learning rate to decay to in the learning rate scheduler.
        """
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=init_lr,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0),
        )
        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=kwargs.get("T_max", 1000),
            eta_min=kwargs.get("eta_min", 3e-6),
        )

        patience = stopper_patience
        self.stopper = EarlyStopper(patience=patience, min_diff=3e-1)
        self.save_path = save_path
        self.save_every = save_every
        self.data_size = len(model.data)

    def score_train_data(self, minibatch: MinibatchSample) -> float:
        return self.model(minibatch)

    @torch.no_grad()
    def score_test_data(self, test_data: Dataset) -> float:
        if False:
            ...
        # commenting the following line, since it depends on code that I do not have.
        # See imports.
        # if isinstance(self.model, CovariateTM):
        # return -self.model.score(test_data.x, test_data.response).mean().item()
        else:
            return -self.model.score(test_data.response).mean().item()

    def fit_model(
        self,
        train_data: Dataset,
        test_data: Dataset,
        num_epochs: int,
        batch_size: int | None = None,
        model_name: str | None = None,
    ) -> np.ndarray:
        """Fits a model to data using test_data for validation.

        batch_size unused in this model. Included for compatibility with other
        models.
        """
        tracked_loss = np.zeros((num_epochs, 2))

        if batch_size is None:
            batch_size = self.data_size
        normalconst = self.data_size / batch_size

        sample_size = train_data.response.shape[1]

        device = next(self.model.parameters()).device

        tqdm_range = tqdm(range(num_epochs))
        for epoch in tqdm_range:
            batch_indices = self.get_batch_index(batch_size, device=device)
            for batch in batch_indices:
                self.optimizer.zero_grad()
                minibatch = train_data[batch]

                loss = normalconst * self.model(minibatch)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                train_loss = self.score_train_data(train_data[:]) / sample_size
                # NLL is averaged over the test set in score_test_data
                nll = self.score_test_data(test_data)
                tracked_loss[epoch, 0] = train_loss
                tracked_loss[epoch, 1] = nll

            if self.scheduler:
                self.scheduler.step()

            if self.stopper:
                if self.stopper.step(tracked_loss[epoch, 1], self.model.state_dict()):
                    self.model.load_state_dict(self.stopper.best_state)
                    break

            self.save_model(epoch, model_name=model_name)
            desc = "Epoch {} | Train loss {:.4f} | Test loss {:.4f}".format(
                epoch, *tracked_loss[epoch]
            )
            tqdm_range.set_description(desc)

        tracked_loss[epoch:] = np.nan  # type: ignore
        return tracked_loss

# Commented out, because it depends on code that I do not have.
# class VariationalModelRunner(Runner):
#     """Model runner for variational models requiring dual optimizers.

#     This runner separates the kernel and variational parameters into separate
#     optimizers and learning rate schedules.

#     Attributes:
#         model (cm.CovariateTM): The model to fit.
#         kernel_opt (torch.optim.AdamW): The optimizer to use for training the
#             kernel parameters.
#         kernel_schedule (torch.optim.lr_scheduler.CosineAnnealingLR): The learning
#             rate scheduler to use for training the kernel parameters.
#         var_opt (torch.optim.AdamW): The optimizer to use for training the
#             variational parameters.
#         var_schedule (torch.optim.lr_scheduler.CosineAnnealingLR): The learning
#             rate scheduler to use for training the variational parameters.
#         stopper (EarlyStopper | None): The early stopping criterion to use for training.
#         save_path (Path | str | None): The path to save checkpoints to.
#         save_every (int | None): The number of epochs between checkpoints.
#     """

#     def __init__(
#         self,
#         model: CovariateTM,
#         init_lrs: list[float] = [4e-3, 1e-2],
#         stopper_patience: int = 1000,
#         save_path: Path | str | None = None,
#         save_every: int | None = None,
#         **kwargs,
#     ):
#         """Instantiates the runner class to fit a transport map.

#         Arguments:
#         ----------
#         model: TransportMap
#             Any transport map in the batram package.

#         init_lrs: list[float]
#             The initial learning rate to initialize AdamW with for each kernel.
#             Typically it is better to set the learning rate for the kernel
#             hyperparameters quite small compared to the variational parameters.
#             The default values are a good starting point.

#         stopper_patience: int
#             The number of epochs to wait before stopping the optimization.

#         save_path: Path | str | None
#             The path to save checkpoints to.

#         save_every: int | None
#             The number of epochs between checkpoints.

#         kwargs:
#         -------
#         keyword arguments to pass to the optimizer and learning rate scheduler.
#         The optimizer is AdamW and the learning rate scheduler is
#         CosineAnnealingLR from pytorch.

#         AdamW args:
#         betas: tuple (default (0.9, 0.999))
#             The betas to use for the AdamW optimizer.

#         eps: float (default 1e-8)
#             The epsilon to use for the AdamW optimizer.

#         weight_decay: float (default 0.0)
#             The weight decay to use for the AdamW optimizer.


#         CosineAnnealingLR args.
#         T_max: int (default 1000)
#             The number of steps the learning rate schedule should apply for

#         eta_min: float (default 3e-6)
#             The minimum learning rate to decay to in the learning rate scheduler.
#         """
#         self.model = model

#         self.kernel_optimizer = torch.optim.AdamW(
#             list(model.log_mean.parameters())
#             + list(model.response_scale.parameters())
#             + list(model.covar_scale.parameters())
#             + list(model.mean_kernel.parameters())
#             + list(model.var_kernel.parameters()),
#             lr=init_lrs[0],
#             betas=kwargs.get("betas", (0.9, 0.999)),
#             eps=kwargs.get("eps", 1e-8),
#             weight_decay=kwargs.get("weight_decay", 0),
#         )
#         self.kernel_scheduler = CosineAnnealingLR(
#             optimizer=self.kernel_optimizer,
#             T_max=kwargs.get("T_max", 1000),
#             eta_min=kwargs.get("eta_min", 3e-6),
#         )

#         self.variational_optimizer = torch.optim.AdamW(
#             model.variational_lambdas.parameters(),
#             lr=init_lrs[1],
#             betas=kwargs.get("betas", (0.9, 0.999)),
#             eps=kwargs.get("eps", 1e-8),
#             weight_decay=kwargs.get("weight_decay", 0),
#         )
#         self.variational_scheduler = CosineAnnealingLR(
#             optimizer=self.variational_optimizer,
#             T_max=kwargs.get("T_max", 1000),
#             eta_min=kwargs.get("eta_min", 3e-6),
#         )

#         patience = stopper_patience
#         self.stopper = EarlyStopper(patience=patience, min_diff=3e-1)
#         self.save_path = save_path
#         self.save_every = save_every
#         self.data_size = len(model.data)

#     def score_train_data(self, minibatch: MinibatchSample) -> float:
#         return self.model(minibatch)

#     @torch.no_grad()
#     def score_test_data(self, test_data: Dataset) -> float:
#         return -self.model.score(test_data.x, test_data.response).mean().item()

#     def fit_model(
#         self,
#         train_data: Dataset,
#         test_data: Dataset,
#         num_epochs: int,
#         batch_size: int | None = None,
#         model_name: str | None = None,
#     ) -> np.ndarray:
#         """Fits a model to data using test_data for validation."""

#         tracked_loss = np.zeros((num_epochs, 2))

#         if batch_size is None:
#             batch_size = len(train_data)
#         normalconst = self.data_size / batch_size
#         sample_size = train_data.response.shape[1]

#         device = next(self.model.parameters()).device

#         tqdm_range = tqdm(range(num_epochs))
#         for epoch in tqdm_range:
#             batch_indices = self.get_batch_index(batch_size, device=device)
#             for batch in batch_indices:
#                 minibatch = train_data[batch]
#                 self.kernel_optimizer.zero_grad()
#                 self.variational_optimizer.zero_grad()

#                 loss = normalconst * self.model(minibatch)
#                 loss.backward()
#                 self.kernel_optimizer.step()
#                 self.variational_optimizer.step()

#             with torch.no_grad():
#                 try:
#                     elbo = self.model(train_data[:]).item() / sample_size
#                     # NLL is averaged over the test set already in score_test_data
#                     nll = self.score_test_data(test_data)
#                     tracked_loss[epoch, 0] = elbo
#                     tracked_loss[epoch, 1] = nll
#                 except CholeskyDecompositionError:
#                     msg = "An error halted the fit. Returning best model."
#                     warnings.warn(msg)
#                     tracked_loss[epoch, :] = np.nan
#                     self.model.load_state_dict(self.stopper.best_state)
#                     return tracked_loss
#                 except RuntimeError:
#                     msg = "An error halted the fit. Returning best model."
#                     warnings.warn(msg)
#                     tracked_loss[epoch, :] = np.nan
#                     self.model.load_state_dict(self.stopper.best_state)
#                     return tracked_loss

#             self.kernel_scheduler.step()
#             self.variational_scheduler.step()

#             if self.stopper:
#                 if self.stopper.step(tracked_loss[epoch, 1], self.model.state_dict()):
#                     self.model.load_state_dict(self.stopper.best_state)
#                     break

#             self.save_model(epoch, model_name=model_name)
#             desc = "Epoch {} | Train loss {:.4f} | Test loss {:.4f}".format(
#                 epoch, *tracked_loss[epoch]
#             )
#             tqdm_range.set_description(desc)

#         tracked_loss[epoch:] = np.nan  # type: ignore
#         return tracked_loss
