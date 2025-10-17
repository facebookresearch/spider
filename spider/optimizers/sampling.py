"""Define functions to get noise schedule for the optimizer.

Convention:
- All info should be numpy array.

Author: Chaoyi Pan
Date: 2025-08-10
"""

from __future__ import annotations

import loguru
import numpy as np
import torch
import torch.nn.functional as F

from spider.config import Config
from spider.interp import interp


def _sample_ctrls_impl(
    config, ctrls: torch.Tensor, sample_params: dict | None = None
) -> torch.Tensor:
    """Sample control actions from the control signal (implementation).

    Args:
        config: Config
        ctrls: Control actions, shape (horizon_steps, nu)
        sample_params: Optional dict with sampling parameters (e.g., global_noise_scale)

    Returns:
        Control actions, shape (num_samples, horizon_steps, nu)
    """
    # decode sample_params
    global_noise_scale = sample_params.get("global_noise_scale", 1.0)
    # sample knot with shape (num_samples, num_knots, nu)
    knot_samples = (
        torch.randn_like(config.noise_scale, device=config.device)
        * config.noise_scale
        * global_noise_scale
    )
    # interp to horizon_steps
    delta_ctrl_samples = interp(knot_samples, config.knot_steps)
    # add to ctrls
    ctrls_samples = ctrls + delta_ctrl_samples
    return ctrls_samples


# Compiled version
_sample_ctrls_compiled = torch.compile(_sample_ctrls_impl)


def sample_ctrls(
    config, ctrls: torch.Tensor, sample_params: dict | None = None
) -> torch.Tensor:
    """Sample control actions from the control signal.

    Args:
        config: Config
        ctrls: Control actions, shape (horizon_steps, nu)
        sample_params: Optional dict with sampling parameters (e.g., global_noise_scale)

    Returns:
        Control actions, shape (num_samples, horizon_steps, nu)
    """
    if config.use_torch_compile:
        return _sample_ctrls_compiled(config, ctrls, sample_params)
    else:
        return _sample_ctrls_impl(config, ctrls, sample_params)


def make_rollout_fn(
    step_env,
    save_state,
    load_state,
    get_reward,
    get_terminal_reward,
    get_trace,
    save_env_params,
    load_env_params,
):
    def rollout(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_param: dict,
    ) -> torch.Tensor:
        """Rollout the control actions to get reward

        Args:
            config: Config
            env: Environment
            ctrls: Control actions, shape (horizon_steps, nu)
            ref_slice: Reference slice, shape (nq, nv, nu, ncon, ncon_pos)

        Returns:
            Reward, shape (num_samples,)
            Info: dict, including trace (N, H, n_trace, 3)
        """
        # save initial state
        init_state = save_state(env)

        # save initial env params (active group pointer)
        init_env_param = save_env_params(config, env)

        # select rollout graph/data pointers for this rollout
        env = load_env_params(config, env, env_param)

        # rollout to get reward
        N, H = ctrls.shape[:2]
        trace_list = []
        cum_rew = torch.zeros(N, device=config.device)
        info_list = []
        for t in range(H):
            # step the environment
            step_env(config, env, ctrls[:, t])  # (N, nu)
            # get reward
            ref = [r[t] for r in ref_slice]
            rew, info = (
                get_reward(config, env, ref)
                if t < H - 1
                else get_terminal_reward(config, env, ref)
            )
            cum_rew += rew
            # get trace
            trace = get_trace(config, env)
            trace_list.append(trace)
            info_list.append(info)
        info_combined = {
            k: torch.stack([info[k] for info in info_list], axis=0)
            for k in info_list[0].keys()
        }
        mean_info = {k: v.mean(axis=0) for k, v in info_combined.items()}
        mean_rew = cum_rew / H

        # reset all envs back to initial state
        env = load_state(env, init_state)

        # reset env params
        env = load_env_params(config, env, init_env_param)

        # get info
        trace_list = torch.stack(trace_list, dim=1)
        info = {
            "trace": trace_list,  # (N, H, n_trace, 3)
            **mean_info,
        }

        return mean_rew, info

    return rollout


def _compute_weights_impl(
    rews: torch.Tensor, num_samples: int, temperature: float
) -> torch.Tensor:
    """Compute softmax weights from rewards (implementation).

    Args:
        rews: Rewards, shape (num_samples,)
        num_samples: Number of samples
        temperature: Temperature for softmax

    Returns:
        Weights, shape (num_samples,)
    """
    # Handle NaNs and compute weights over N samples
    nan_mask = torch.isnan(rews) | torch.isinf(rews)
    rews_min = (
        rews[~nan_mask].min()
        if (~nan_mask).any()
        else torch.tensor(-1000.0, device=rews.device)
    )
    rews = torch.where(nan_mask, rews_min, rews)

    # Select top 10% samples for softmax weighting
    top_k = max(1, int(0.1 * num_samples))
    top_indices = torch.topk(rews, k=top_k, largest=True).indices

    # Initialize weights as zeros and compute softmax only for top samples
    weights = torch.zeros_like(rews)
    top_rews = rews[top_indices]
    top_rews_normalized = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
    top_weights = F.softmax(top_rews_normalized / temperature, dim=0)
    weights[top_indices] = top_weights

    return weights, nan_mask


# Compiled version
_compute_weights_compiled = torch.compile(_compute_weights_impl)


def make_optimize_once_fn(
    rollout,
):
    def optimize_once(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_params: list[dict] = [{}],
        sample_params: dict | None = None,
    ) -> torch.Tensor:
        """Single step optimization of the policy parameters using DIAL MPC, no annealing is involved

        Args:
            config: Config
            graph: Warp graph
            model_wp: Warp model
            data_wp: Warp data
            ctrls: Control actions, shape (horizon_steps, num_actions)

        Returns:
            Control actions, shape (horizon_steps, num_actions)
        """
        # sample ctrls
        ctrls_samples = sample_ctrls(
            config, ctrls, sample_params
        )  # (num_samples, horizon_steps, num_actions)

        # rollout
        # domain randomization: pick the minimum reward across all DR parameter sets
        min_rew = torch.full((config.num_samples,), float("inf"), device=config.device)
        for env_param in env_params:
            rews, rollout_info = rollout(
                config,
                env,
                ctrls_samples,
                ref_slice,
                env_param,
            )
            min_rew = torch.minimum(min_rew, rews)
        # Use worst-case rewards across DR parameter sets
        rews = min_rew

        # Compute weights using compiled or non-compiled version
        if config.use_torch_compile:
            weights, nan_mask = _compute_weights_compiled(
                rews, config.num_samples, config.temperature
            )
        else:
            weights, nan_mask = _compute_weights_impl(
                rews, config.num_samples, config.temperature
            )

        if nan_mask.any():
            loguru.logger.warning(
                f"NaNs or infs in rews: {nan_mask.sum()}/{config.num_samples}"
            )

        ctrls_mean = (weights[:, None, None] * ctrls_samples).sum(dim=0)

        # down sample traces by selecting topk and uniform samples for visualization
        n_uni = max(0, min(config.num_trace_uniform_samples, config.num_samples))
        n_topk = max(0, min(config.num_trace_topk_samples, config.num_samples))
        idx_uni = (
            torch.linspace(
                0,
                config.num_samples - 1,
                steps=n_uni,
                dtype=torch.long,
                device=config.device,
            )
            if n_uni > 0
            else torch.tensor([], dtype=torch.long, device=config.device)
        )
        idx_top = (
            torch.topk(rews, k=n_topk, largest=True).indices
            if n_topk > 0
            else torch.tensor([], dtype=torch.long, device=config.device)
        )
        sel_idx = torch.cat([idx_uni, idx_top], dim=0).long()

        # compute info
        info = {}
        for k, v in rollout_info.items():
            if k not in ["trace", "trace_sample"]:
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                if v.ndim == 1:
                    k_max = k + "_max"
                    k_min = k + "_min"
                    k_median = k + "_median"
                    k_mean = k + "_mean"
                    info[k_max] = v.max()
                    info[k_min] = v.min()
                    info[k_median] = np.median(v)
                    info[k_mean] = v.mean()
        # log reward
        rews_np = rews.cpu().numpy()
        improvement = rews_np.max() - rews_np[0]
        info["improvement"] = improvement
        info["rew_max"] = rews_np.max()
        info["rew_min"] = rews_np.min()
        info["rew_median"] = np.median(rews_np)
        info["rew_mean"] = rews_np.mean()

        # Downsample and store trace site positions for selected sample trajectories
        if "trace" in rollout_info:
            info["trace_sample"] = (
                rollout_info["trace"][sel_idx].cpu().numpy()
            )  # (M, H, n_trace, 3)

        return ctrls_mean, info

    return optimize_once


def make_optimize_fn(
    optimize_once,
):
    def optimize(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
    ):
        """Full optimization loop at certain time step. Mainly involves simulation parameter annealing and sampling parameter annealing"""
        infos = []

        # schedule sampling parameters
        sample_params_list = []
        for i in range(config.max_num_iterations):
            sample_params = {
                "global_noise_scale": config.beta_traj**i,
            }
            sample_params_list.append(sample_params)

        # optimize
        improvement_history = []
        for i in range(config.max_num_iterations):
            ctrls, info = optimize_once(
                config,
                env,
                ctrls,
                ref_slice,
                config.env_params_list[i],
                sample_params_list[i],
            )
            infos.append(info)
            improvement_history.append(info["improvement"])

            # early stopping: check if last n steps all have improvement below threshold
            if len(improvement_history) >= config.improvement_check_steps:
                recent_improvements = improvement_history[
                    -config.improvement_check_steps :
                ]
                if all(
                    imp < config.improvement_threshold for imp in recent_improvements
                ):
                    break

        # TODO: think about a better logic
        # append zeros to infos to make sure the length is the same as max_num_iterations
        fake_info = {}
        for k, v in infos[0].items():
            fake_info[k] = np.zeros_like(v)
        for _ in range(config.max_num_iterations - len(infos)):
            infos.append(fake_info)
        info_aggregated = {}
        for k in infos[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in infos], axis=0)
        info_aggregated["opt_steps"] = np.array([i + 1])
        return ctrls, info_aggregated

    return optimize
