from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from liesel.goose.interface import LieselInterface
from liesel.goose.optim import (
    OptimResult,
    Stopper,
    _generate_batch_indices,
    _validate_log_prob_decomposition,
)
from liesel.goose.types import Array, Position
from liesel.model import Model, Var
from tqdm import tqdm


def optim_loc_batched(
    model: Model,
    response_train: Var,
    locs: Var,
    params: Sequence[str],
    optimizer: optax.GradientTransformation | None = None,
    stopper: Stopper | None = None,
    loc_batch_size: int | None = None,
    batch_seed: int | None = None,
    save_position_history: bool = True,
    response_validation: Var | None = None,
    restore_best_position: bool = True,
    prune_history: bool = True,
    n_train: int | None = None,
    n_validation: int | None = None,
) -> OptimResult:
    # ---------------------------------------------------------------------------------
    # Validation input
    if restore_best_position:
        assert (
            save_position_history
        ), "Cannot restore best position if history is not saved."

    # ---------------------------------------------------------------------------------
    # Pre-process inputs

    batch_seed = (
        batch_seed if batch_seed is not None else np.random.randint(low=1, high=1000)
    )

    if stopper is None:
        stopper = Stopper(max_iter=100, patience=10)

    user_patience = stopper.patience

    if response_validation is None:
        response_validation = response_train
        stopper.patience = stopper.max_iter

    if optimizer is None:
        optimizer = optax.adam(learning_rate=1e-2)

    nloc = locs.value.shape[0]

    batch_size = loc_batch_size if loc_batch_size is not None else nloc

    interface = LieselInterface(model)
    position = interface.extract_position(params, model.state)
    interface._model.auto_update = False

    # ---------------------------------------------------------------------------------
    # Validate model log prob decomposition
    _validate_log_prob_decomposition(interface, position=position, state=model.state)

    # ---------------------------------------------------------------------------------
    # Define loss function(s)

    n_train = jnp.shape(response_train.value)[1]
    n_validation = jnp.shape(response_validation.value)[1]

    likelihood_scalar_validation = n_train / n_validation
    n_batches = nloc // batch_size

    def _neg_log_prob_batch_train(
        position: Position, response_train: Var, batch_indices: Array
    ):
        batched_response = {response_train.name: response_train.value[batch_indices, :]}
        batched_location = {locs.name: locs.value[batch_indices, ...]}
        position = position | batched_response | batched_location  # type: ignore

        updated_state = interface.update_state(position, model.state)
        log_lik = updated_state["_model_log_lik"].value
        log_prior = updated_state["_model_log_prior"].value
        neg_log_prob_train = -(log_lik + log_prior)

        return neg_log_prob_train

    def _neg_log_prob_batch(
        position: Position,
        response_train: Var,
        response_validation: Var,
        batch_indices: Array,
    ):
        batched_response = {response_train.name: response_train.value[batch_indices, :]}
        batched_location = {locs.name: locs.value[batch_indices, ...]}
        position = position | batched_response | batched_location  # type: ignore

        updated_state = interface.update_state(position, model.state)
        log_lik = updated_state["_model_log_lik"].value
        log_prior = updated_state["_model_log_prior"].value
        neg_log_prob_train = -(log_lik + log_prior)

        batched_response_validation = {
            response_validation.name: response_validation.value[batch_indices, :]
        }
        updated_state = interface.update_state(
            batched_response_validation, updated_state
        )
        log_lik = likelihood_scalar_validation * updated_state["_model_log_lik"].value
        log_prior = updated_state["_model_log_prior"].value
        neg_log_prob_validation = -(log_lik + log_prior)

        return neg_log_prob_train, neg_log_prob_validation

    def _neg_log_prob(
        position: Position,
        response_train: Var,
        response_validation: Var,
        batches: Array,
    ):
        neg_log_prob_train_batches = jnp.empty(shape=(n_batches,))
        neg_log_prob_validation_batches = jnp.empty(shape=(n_batches,))

        def body_fun(i, val):
            neg_log_prob_train_batches, neg_log_prob_validation_batches, batches = val
            nlp_i_train, nlp_i_validation = _neg_log_prob_batch(
                position=position,
                response_train=response_train,
                response_validation=response_validation,
                batch_indices=batches[i],
            )
            neg_log_prob_train_batches = neg_log_prob_train_batches.at[i].set(
                nlp_i_train
            )
            neg_log_prob_validation_batches = neg_log_prob_validation_batches.at[i].set(
                nlp_i_validation
            )
            return neg_log_prob_train_batches, neg_log_prob_validation_batches, batches

        init_val = (
            neg_log_prob_train_batches,
            neg_log_prob_validation_batches,
            batches,
        )

        neg_log_prob_train_batches, neg_log_prob_validation_batches, _ = (
            jax.lax.fori_loop(
                lower=0, upper=n_batches, body_fun=body_fun, init_val=init_val
            )
        )

        return jnp.sum(neg_log_prob_train_batches), jnp.sum(
            neg_log_prob_validation_batches
        )

    neg_log_prob_grad_batch = jax.grad(_neg_log_prob_batch_train, argnums=0)

    # ---------------------------------------------------------------------------------
    # Initialize history

    history: dict[str, Any] = dict()
    history["loss_train"] = jnp.zeros(shape=stopper.max_iter)
    history["loss_validation"] = jnp.zeros(shape=stopper.max_iter)

    if save_position_history:
        history["position"] = {
            name: jnp.zeros((stopper.max_iter,) + jnp.shape(value))
            for name, value in position.items()
        }
        history["position"] = jax.tree.map(
            lambda d, pos: d.at[0].set(pos), history["position"], position
        )
    else:
        history["position"] = None

    batches = _generate_batch_indices(
        key=jax.random.PRNGKey(batch_seed), n=nloc, batch_size=batch_size
    )
    loss_train_start, loss_validation_start = _neg_log_prob(
        position=position,
        response_train=response_train,
        response_validation=response_validation,
        batches=batches,
    )
    history["loss_train"] = history["loss_train"].at[0].set(loss_train_start)
    history["loss_validation"] = (
        history["loss_validation"].at[0].set(loss_validation_start)
    )

    # ---------------------------------------------------------------------------------
    # Initialize while loop carry dictionary

    init_val: dict[str, Any] = dict()
    init_val["while_i"] = 0
    init_val["history"] = history
    init_val["position"] = position
    init_val["opt_state"] = optimizer.init(position)
    init_val["key"] = jax.random.PRNGKey(batch_seed)

    # ---------------------------------------------------------------------------------
    # Define while loop body
    progress_bar = tqdm(
        total=stopper.max_iter - 1,
        desc=(
            f"Training loss: {loss_train_start:.3f}, Validation loss:"
            f" {loss_validation_start:.3f}"
        ),
        position=0,
        leave=True,
    )

    def tqdm_callback(val):
        i = val["while_i"]
        loss_train = val["history"]["loss_train"][i]
        loss_validation = val["history"]["loss_validation"][i]
        desc = (
            f"Training loss: {loss_train:.3f}, Validation loss: {loss_validation:.3f}"
        )
        progress_bar.update(1)
        progress_bar.set_description(desc)

    def body_fun(val: dict):
        _, subkey = jax.random.split(val["key"])
        batches = _generate_batch_indices(key=subkey, n=nloc, batch_size=batch_size)

        # -----------------------------------------------------------------------------
        # Loop over batches

        def _fori_body(i, val):
            batch = batches[i]
            pos = val["position"]
            grad = neg_log_prob_grad_batch(
                pos, response_train=response_train, batch_indices=batch
            )
            updates, opt_state = optimizer.update(grad, val["opt_state"], params=pos)
            val["position"] = optax.apply_updates(pos, updates)
            val["opt_state"] = opt_state

            return val

        val = jax.lax.fori_loop(
            body_fun=_fori_body, init_val=val, lower=0, upper=len(batches)
        )

        # -----------------------------------------------------------------------------
        # Save values and increase counter
        val["while_i"] += 1

        loss_train, loss_validation = _neg_log_prob(
            val["position"],
            response_train=response_train,
            response_validation=response_validation,
            batches=batches,
        )

        val["history"]["loss_train"] = (
            val["history"]["loss_train"].at[val["while_i"]].set(loss_train)
        )
        val["history"]["loss_validation"] = (
            val["history"]["loss_validation"].at[val["while_i"]].set(loss_validation)
        )

        if save_position_history:
            pos_hist = val["history"]["position"]
            val["history"]["position"] = jax.tree.map(
                lambda d, pos: d.at[val["while_i"]].set(pos), pos_hist, val["position"]
            )

        jax.debug.callback(tqdm_callback, val)
        return val

    # ---------------------------------------------------------------------------------
    # Run while loop

    val = jax.lax.while_loop(
        cond_fun=lambda val: stopper.continue_(
            val["while_i"], val["history"]["loss_validation"]
        ),
        body_fun=body_fun,
        init_val=init_val,
    )

    max_iter = val["while_i"]

    # ---------------------------------------------------------------------------------
    # Set final position and model state
    stopper.patience = user_patience
    ibest = stopper.which_best_in_recent_history(
        i=max_iter, loss_history=val["history"]["loss_validation"]
    )

    if restore_best_position:
        final_position: Position = {
            name: pos[ibest] for name, pos in val["history"]["position"].items()
        }  # type: ignore
    else:
        final_position = val["position"]

    final_state = interface.update_state(final_position, model.state)

    # ---------------------------------------------------------------------------------
    # Set unused values in history to nan

    val["history"]["loss_train"] = (
        val["history"]["loss_train"].at[(max_iter + 1) :].set(jnp.nan)
    )
    val["history"]["loss_validation"] = (
        val["history"]["loss_validation"].at[(max_iter + 1) :].set(jnp.nan)
    )
    if save_position_history:
        for name, value in val["history"]["position"].items():
            val["history"]["position"][name] = value.at[(max_iter + 1) :, ...].set(
                jnp.nan
            )

    # ---------------------------------------------------------------------------------
    # Remove unused values in history, if applicable

    if prune_history:
        val["history"]["loss_train"] = val["history"]["loss_train"][: (max_iter + 1)]
        val["history"]["loss_validation"] = val["history"]["loss_validation"][
            : (max_iter + 1)
        ]
        if save_position_history:
            for name, value in val["history"]["position"].items():
                val["history"]["position"][name] = value[: (max_iter + 1), ...]

    # ---------------------------------------------------------------------------------
    # Initialize results object and return

    result = OptimResult(
        model_state=final_state,
        position=final_position,
        iteration=max_iter,
        iteration_best=ibest,
        history=val["history"],
        max_iter=stopper.max_iter,
        n_train=n_train,
        n_validation=n_validation,
    )

    return result
