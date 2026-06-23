from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.value_head import ValueHead


def _as_bool(x, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}



class MotusPolicy(nn.Module, BasePolicy):
    """RLinf eval-only adapter for official Motus RoboTwin policy.

    This adapter deliberately reuses the official Motus RoboTwin policy wrapper
    in /root/autodl-tmp/RoboTwin/policy/Motus/deploy_policy.py.

    First target:
      - eval only
      - RoboTwin
      - batch size 1
      - actions shape [B, 16, 14]
    """

    def __init__(self, cfg, torch_dtype=None):
        super().__init__()
        self.cfg = cfg
        self.torch_dtype = torch_dtype
        self.model_type = "motus"

        motus_cfg = cfg.get("motus", {})
        self.policy_path = Path(str(motus_cfg.get("policy_path"))).expanduser()
        self.checkpoint_path = str(motus_cfg.get("checkpoint_path", cfg.model_path))
        self.wan_path = str(motus_cfg.get("wan_path"))
        self.vlm_path = str(motus_cfg.get("vlm_path"))
        self.config_path = str(motus_cfg.get("config_path"))

        self.num_action_chunks = int(cfg.get("num_action_chunks", 16))
        self.action_dim = int(cfg.get("action_dim", 14))
        self.allow_batch_size = int(motus_cfg.get("allow_batch_size", 1))
        self.save_predicted_frames = bool(motus_cfg.get("save_predicted_frames", False))
        self.num_inference_timesteps = int(motus_cfg.get("num_inference_timesteps", 10))
        self.batch_inference = _as_bool(motus_cfg.get("batch_inference", False), default=False)
        self.decode_video = _as_bool(motus_cfg.get("decode_video", False), default=False)
        self._warned_decode_video_unsupported = False
        self._train_batch_print_count = 0
        self.scene_prefix = str(
            motus_cfg.get(
                "scene_prefix",
                "The whole scene is in a realistic, industrial art style with three views: "
                "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                "The aloha robot is currently performing the following task: ",
            )
        )

        self.trainable = str(motus_cfg.get("trainable", "none")).lower()
        self.freeze_video_model = _as_bool(motus_cfg.get("freeze_video_model", True), default=True)
        self.freeze_vlm_model = _as_bool(motus_cfg.get("freeze_vlm_model", True), default=True)
        self.freeze_und_expert = _as_bool(motus_cfg.get("freeze_und_expert", True), default=True)
        self.freeze_t5_encoder = _as_bool(motus_cfg.get("freeze_t5_encoder", True), default=True)
        self.logprob_mode = str(motus_cfg.get("logprob_mode", "transition_velocity")).lower()
        self.logprob_sigma = float(motus_cfg.get("logprob_sigma", 1.0))
        self.collect_denoise_step = str(motus_cfg.get("collect_denoise_step", "random")).lower()
        self.ignore_first = _as_bool(motus_cfg.get("ignore_first", False), default=False)
        self.ignore_last = _as_bool(motus_cfg.get("ignore_last", False), default=False)
        self.add_value_head = _as_bool(cfg.get("add_value_head", False), default=False)
        self.detach_critic_input = _as_bool(motus_cfg.get("detach_critic_input", True), default=True)
        self.value_feature = str(motus_cfg.get("value_feature", "action_tokens")).lower()
        if self.value_feature != "action_tokens":
            raise ValueError(
                f"Unsupported Motus value_feature={self.value_feature!r}. "
                "OpenPI-aligned PPO currently uses action_tokens."
            )

        if not self.policy_path.exists():
            raise FileNotFoundError(f"Motus policy_path not found: {self.policy_path}")
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Motus checkpoint_path not found: {self.checkpoint_path}")
        if not Path(self.wan_path).exists():
            raise FileNotFoundError(f"Motus wan_path not found: {self.wan_path}")
        if not Path(self.vlm_path).exists():
            raise FileNotFoundError(f"Motus vlm_path not found: {self.vlm_path}")
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Motus config_path not found: {self.config_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        deploy_module = self._load_official_deploy_policy()
        OfficialMotusPolicy = getattr(deploy_module, "MotusPolicy")

        self._policy = OfficialMotusPolicy(
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            wan_path=self.wan_path,
            vlm_path=self.vlm_path,
            device=self.device,
            log_dir=None,
            task_name="rlinf_eval",
        )

        # Official wrapper saves predicted frame grids by default. Disable for
        # RLinf eval unless explicitly requested.
        self._policy.save_images = self.save_predicted_frames

        # Expose official Motus nn.Module as a submodule so RLinf Actor/FSDP/optimizer/checkpoint can see it.
        self.model = self._policy.model
        if self.add_value_head:
            value_input_dim = int(getattr(self.model.config, "action_expert_dim", 1024))
            self.value_head = ValueHead(
                input_dim=value_input_dim,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            ).to(self.device)
        self._configure_trainable_parameters()

        # Make YAML-level inference-step override effective. The official wrapper
        # reads this value from config_dict inside get_action().
        try:
            self._policy.config_dict["model"]["inference"][
                "num_inference_timesteps"
            ] = self.num_inference_timesteps
        except Exception:
            pass

    def _load_official_deploy_policy(self):
        deploy_path = self.policy_path / "deploy_policy.py"
        if not deploy_path.exists():
            raise FileNotFoundError(f"deploy_policy.py not found: {deploy_path}")

        # deploy_policy.py imports top-level packages `models`, `utils`, `wan`.
        # Put official policy dir at the front so these resolve to clean Motus files.
        policy_str = str(self.policy_path.resolve())
        models_str = str((self.policy_path / "models").resolve())
        for p in [models_str, policy_str]:
            if p not in sys.path:
                sys.path.insert(0, p)

        module_name = "rlinf_external_motus_deploy_policy"
        spec = importlib.util.spec_from_file_location(module_name, str(deploy_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load official Motus deploy_policy.py from {deploy_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _configure_trainable_parameters(self) -> None:
        """Set requires_grad flags for RLinf training.

        Optimizer is created by RLinf Actor/FSDP, not here.
        """
        if not hasattr(self, "model"):
            return

        # Default: freeze everything unless a training mode explicitly enables it.
        for p in self.model.parameters():
            p.requires_grad_(False)

        if self.trainable in {"none", "eval", "false"}:
            trainable = 0
        elif self.trainable == "action_expert":
            for p in self.model.action_expert.parameters():
                p.requires_grad_(True)
            trainable = sum(p.numel() for p in self.model.action_expert.parameters() if p.requires_grad)
        elif self.trainable == "action_und":
            for p in self.model.action_expert.parameters():
                p.requires_grad_(True)
            for p in self.model.und_expert.parameters():
                p.requires_grad_(True)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        elif self.trainable == "all":
            for p in self.model.parameters():
                p.requires_grad_(True)
            if self.freeze_video_model and hasattr(self.model, "video_model"):
                for p in self.model.video_model.parameters():
                    p.requires_grad_(False)
            if self.freeze_vlm_model and hasattr(self.model, "vlm_model"):
                for p in self.model.vlm_model.parameters():
                    p.requires_grad_(False)
            if self.freeze_und_expert and hasattr(self.model, "und_expert"):
                for p in self.model.und_expert.parameters():
                    p.requires_grad_(False)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            raise ValueError(
                f"Unknown Motus trainable={self.trainable!r}. "
                "Use one of: none, action_expert, action_und, all."
            )

        self._freeze_fsdp_ignored_fp32_modules()
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(
            f"[RLinf-Motus] trainable={self.trainable}, "
            f"trainable_params={trainable / 1e6:.2f}M",
            flush=True,
        )

    def _freeze_fsdp_ignored_fp32_modules(self) -> None:
        """Freeze small fp32 time modules so FSDP can ignore them safely.

        Motus keeps action time embedding/projection in fp32 for numerical stability.
        FSDP flatten requires uniform dtype in a managed handle, so we keep those
        modules out of FSDP and do not optimize them in the first smoke version.
        """
        if not hasattr(self, "model"):
            return

        action_expert = getattr(self.model, "action_expert", None)
        if action_expert is None:
            return

        for module_name in ["time_embedding", "time_projection"]:
            module = getattr(action_expert, module_name, None)
            if module is None:
                continue
            module.eval()
            for p in module.parameters(recurse=True):
                p.requires_grad_(False)

    def get_fsdp_ignored_modules(self):
        """Modules excluded from FSDP flatten.

        Frozen large towers are ignored to avoid unnecessary flattening.
        fp32 time modules are ignored to avoid bf16/fp32 mixed flatten groups.
        """
        modules = []

        if not hasattr(self, "model"):
            return modules

        for module_name in ["video_model", "vlm_model", "und_expert"]:
            module = getattr(self.model, module_name, None)
            if module is not None:
                modules.append(module)

        action_expert = getattr(self.model, "action_expert", None)
        if action_expert is not None:
            for module_name in ["time_embedding", "time_projection"]:
                module = getattr(action_expert, module_name, None)
                if module is not None:
                    modules.append(module)

        # Deduplicate while preserving order.
        out = []
        seen = set()
        for module in modules:
            mid = id(module)
            if mid not in seen:
                seen.add(mid)
                out.append(module)
        return out

    def forward(self, *args, **kwargs):
        return self.default_forward(*args, **kwargs)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Actor update forward for RLinf PPO/GRPO.

        Recompute current-policy logprobs for the denoise transition captured
        during rollout. The replay object mirrors OpenPI's chain-based flow RL
        path: action/video chains plus denoise indices define the stochastic
        transition whose logprob is recomputed under the current actor.
        """
        if forward_inputs is None:
            raise ValueError("Motus default_forward requires forward_inputs.")
        if "motus_action_chains" not in forward_inputs or "motus_video_chains" not in forward_inputs:
            raise ValueError(
                "Motus default_forward requires OpenPI-style chain replay fields "
                "`motus_action_chains` and `motus_video_chains`. Regenerate rollouts "
                "with the current Motus adapter instead of using selected-transition-only data."
            )
        if "motus_denoise_inds" not in forward_inputs:
            raise ValueError("Motus default_forward requires `motus_denoise_inds`.")
        if not hasattr(self.model, "get_log_prob_value"):
            raise RuntimeError(
                "Motus model must implement get_log_prob_value() for OpenPI-style "
                "chain replay. Update third_party/Motus/models/motus.py."
            )

        device = next(self.model.parameters()).device
        action_chains = forward_inputs["motus_action_chains"].to(
            device=device,
            dtype=self.model.dtype,
        )
        video_chains = forward_inputs["motus_video_chains"].to(
            device=device,
            dtype=self.model.dtype,
        )
        denoise_inds = forward_inputs["motus_denoise_inds"].to(
            device=device,
            dtype=torch.long,
        )
        state = forward_inputs["motus_state"].to(device=device, dtype=self.model.dtype)
        language_embeddings = forward_inputs["motus_language_embeddings"].to(
            device=device,
            dtype=self.model.dtype,
        )
        vlm_inputs = self._vlm_inputs_from_forward_inputs(forward_inputs, device=device)

        compute_values = bool(kwargs.get("compute_values", False))
        if compute_values and not self.add_value_head:
            raise NotImplementedError(
                "Motus PPO requires actor.model.add_value_head=True. "
                "GRPO/actor-only configs may keep add_value_head=False."
            )

        logprobs, value_features, entropy = self.model.get_log_prob_value(
            video_chains=video_chains,
            action_chains=action_chains,
            denoise_inds=denoise_inds,
            state=state,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
            logprob_mode=self.logprob_mode,
            logprob_sigma=self.logprob_sigma,
            num_action_chunks=self.num_action_chunks,
            action_dim=self.action_dim,
            compute_values=compute_values,
            return_value_features=compute_values,
        )
        values = self._compute_values_from_features(value_features) if compute_values else None

        out = {
            "logprobs": logprobs.float(),
            "values": values,
        }

        if kwargs.get("compute_entropy", False):
            if entropy is None:
                entropy = torch.zeros_like(logprobs, dtype=torch.float32)
            out["entropy"] = entropy.float()

        return out

    def _compute_values_from_features(self, value_features: torch.Tensor | None) -> torch.Tensor:
        """Compute PPO critic values from Motus action-token features.

        OpenPI computes critic values from action suffix features. Motus mirrors
        that by pooling action-expert tokens from the replayed denoise step; the
        third-party Motus model returns those pooled features, while the RLinf
        adapter owns the ValueHead so RLinf/FSDP/optimizer can see it directly.
        """
        if not self.add_value_head or not hasattr(self, "value_head"):
            raise NotImplementedError(
                "Motus value head is not initialized. Set actor.model.add_value_head=True for PPO."
            )
        if value_features is None:
            raise RuntimeError("Motus get_log_prob_value did not return value_features.")

        value_param = next(self.value_head.parameters())
        value_features = value_features.to(
            device=value_param.device,
            dtype=value_param.dtype,
        )
        if self.detach_critic_input:
            value_features = value_features.detach()

        if value_features.dim() == 3:
            # OpenPI rollout computes a value for each denoise step and averages
            # values over the denoise dimension for prev_values.
            bsz, n_steps, hidden = value_features.shape
            flat_values = self.value_head(value_features.reshape(bsz * n_steps, hidden))
            return flat_values.reshape(bsz, n_steps, -1).mean(dim=1)[:, :1].float()
        if value_features.dim() != 2:
            raise ValueError(
                f"Motus value_features must be [B,H] or [B,S,H], got {tuple(value_features.shape)}"
            )
        return self.value_head(value_features).reshape(value_features.shape[0], -1)[:, :1].float()

    def train(self, mode: bool = True):
        super().train(mode)

        if not hasattr(self, "_policy") or not hasattr(self._policy, "model"):
            return self

        if not mode:
            self._policy.model.eval()
            return self

        # Actor/FSDP training mode. Keep frozen/context modules deterministic.
        self._policy.model.train()

        if self.trainable in {"none", "eval", "false"}:
            self._policy.model.eval()
            return self

        for module_name in ["video_model", "vlm_model", "und_expert"]:
            module = getattr(self._policy.model, module_name, None)
            if module is not None:
                module.eval()

        action_expert = getattr(self._policy.model, "action_expert", None)
        if action_expert is not None:
            # Keep policy stochasticity controlled by flow_sde noise, not module
            # train-mode randomness. eval() does not disable gradients; params
            # with requires_grad=True are still optimized in actor default_forward.
            action_expert.eval()

        return self

    def eval(self):
        super().eval()
        if hasattr(self, "_policy") and hasattr(self._policy, "model"):
            self._policy.model.eval()
        return self

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict[str, Any], mode: str = "eval", **kwargs):
        batch_size = self._infer_batch_size(env_obs)
        if batch_size > self.allow_batch_size:
            raise ValueError(
                f"Motus adapter supports batch_size <= {self.allow_batch_size}, "
                f"got {batch_size}. Increase rollout.model.motus.allow_batch_size "
                "or reduce env count per rollout worker."
            )

        if mode not in {"train", "eval"}:
            raise ValueError(f"Motus predict_action_batch mode must be 'train' or 'eval', got {mode!r}")

        # Match OpenPI's train/eval structure: both modes use the same vectorized
        # model inference path; train only asks that path to return replay chains.
        return self._predict_action_batch_vectorized(env_obs, batch_size, mode=mode)

    def _predict_action_batch_loop(
        self, env_obs: dict[str, Any], batch_size: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Fallback path: run official singleton wrapper once per sample."""
        actions = []

        for i in range(batch_size):
            observation = self._build_official_observation(env_obs, i)
            instruction = self._get_instruction(env_obs, i)

            self._reset_official_policy_transient_state()
            self._policy.set_instruction(instruction)
            self._policy.update_obs(observation)
            action_i = self._policy.get_action()

            action_i = self._validate_action_array(action_i, sample_idx=i)
            actions.append(torch.from_numpy(action_i))

        action_tensor = torch.stack(actions, dim=0).float().cpu().contiguous()
        return action_tensor, {"forward_inputs": {}}

    def _predict_action_batch_vectorized(
        self, env_obs: dict[str, Any], batch_size: int, mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """True RLinf batch path: B different env observations -> one Motus inference.

        This is not TTS candidate batching. It does not repeat one observation.
        It stacks B different frames/states/instructions from RLinf env_obs.
        """
        frames: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        t5_list: list[torch.Tensor] = []
        vlm_inputs_list: list[dict[str, torch.Tensor]] = []
        instructions: list[str] = []

        for i in range(batch_size):
            frame_i, state_i, t5_i, vlm_i, instruction_i = self._prepare_one_for_vectorized(
                env_obs, i
            )
            frames.append(frame_i)
            states.append(state_i)
            t5_list.append(t5_i)
            vlm_inputs_list.append(vlm_i)
            instructions.append(instruction_i)

        first_frame = torch.stack(frames, dim=0).to(self.device)
        state = torch.stack(states, dim=0).to(self.device)

        # Train path: sample actions and record one denoise transition for πRL logprob.
        if mode == "train":
            return self._predict_action_batch_vectorized_train(
                first_frame=first_frame,
                state=state,
                t5_list=t5_list,
                vlm_inputs_list=vlm_inputs_list,
                batch_size=batch_size,
            )

        self._ensure_motus_grid_batch_size(batch_size)
        language_embeddings = self._stack_t5_embeddings(t5_list).to(self.device)
        vlm_inputs_batched = self._collate_vlm_inputs(vlm_inputs_list, device=self.device)

        print(
            f"[RLinf-Motus] vectorized batch inference: "
            f"B={batch_size}, first_frame={tuple(first_frame.shape)}, "
            f"state={tuple(state.shape)}, t5={tuple(language_embeddings.shape)}, "
            f"vlm_input_ids={tuple(vlm_inputs_batched['input_ids'].shape)}, "
            f"decode_video={self.decode_video}",
            flush=True,
        )

        call_kwargs = {
            "first_frame": first_frame,
            "state": state,
            "num_inference_steps": self.num_inference_timesteps,
            "language_embeddings": language_embeddings,
            "vlm_inputs": vlm_inputs_batched,
        }

        sig = inspect.signature(self._policy.model.inference_step)
        if "decode_video" not in sig.parameters:
            raise RuntimeError(
                "Motus eval path requires inference_step(decode_video=...) so train/eval "
                "share the same controllable model path. Update third_party/Motus/models/motus.py."
            )
        call_kwargs["decode_video"] = self.decode_video

        out = self._policy.model.inference_step(**call_kwargs)

        if not isinstance(out, tuple) or len(out) < 2:
            raise RuntimeError(
                f"Motus inference_step must return at least "
                f"(predicted_frames, predicted_actions), got type={type(out)!r}"
            )

        predicted_frames, predicted_actions = out[0], out[1]

        if predicted_actions is None:
            raise RuntimeError("Motus inference_step returned predicted_actions=None")

        actions_np = predicted_actions.detach().float().cpu().numpy()
        if actions_np.ndim != 3:
            raise ValueError(
                f"Motus batched actions must be [B,T,D], got shape {actions_np.shape}"
            )
        if actions_np.shape[0] != batch_size:
            raise ValueError(
                f"Motus batch mismatch: expected B={batch_size}, got {actions_np.shape[0]}"
            )

        actions_np = actions_np[:, : self.num_action_chunks, : self.action_dim]

        if actions_np.shape[1] != self.num_action_chunks:
            raise ValueError(
                f"Motus action chunk mismatch: expected {self.num_action_chunks}, "
                f"got {actions_np.shape[1]}"
            )
        if actions_np.shape[2] != self.action_dim:
            raise ValueError(
                f"Motus action dim mismatch: expected {self.action_dim}, "
                f"got {actions_np.shape[2]}"
            )

        action_tensor = torch.from_numpy(actions_np).float().cpu().contiguous()
        return action_tensor, {"forward_inputs": {}}

    def _predict_action_batch_vectorized_train(
        self,
        *,
        first_frame: torch.Tensor,
        state: torch.Tensor,
        t5_list: list[torch.Tensor],
        vlm_inputs_list: list[dict[str, torch.Tensor]],
        batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Train rollout path.

        Returns environment actions plus old logprobs and tensor-only forward_inputs.
        """
        self._ensure_motus_grid_batch_size(batch_size)

        language_embeddings = self._stack_t5_embeddings(t5_list).to(self.device)
        vlm_inputs_batched = self._collate_vlm_inputs(vlm_inputs_list, device=self.device)

        if self._train_batch_print_count < 3:
            print(
                f"[RLinf-Motus] train batch inference: "
                f"B={batch_size}, first_frame={tuple(first_frame.shape)}, "
                f"state={tuple(state.shape)}, t5={tuple(language_embeddings.shape)}, "
                f"vlm_input_ids={tuple(vlm_inputs_batched['input_ids'].shape)}",
                flush=True,
            )
            self._train_batch_print_count += 1

        trace_step = self._select_trace_step(self.num_inference_timesteps)

        sig = inspect.signature(self._policy.model.inference_step)
        required_params = {"return_trace", "trace_step", "logprob_mode", "logprob_sigma"}
        missing_params = required_params.difference(sig.parameters)
        if missing_params:
            raise RuntimeError(
                "Motus model inference_step is missing OpenPI-style trace parameters: "
                f"{sorted(missing_params)}. Update third_party/Motus/models/motus.py; "
                "selected-transition-only or adapter-local fallback is intentionally disabled."
            )

        call_kwargs = dict(
            first_frame=first_frame,
            state=state,
            num_inference_steps=self.num_inference_timesteps,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs_batched,
            return_trace=True,
            trace_step=trace_step,
            logprob_mode=self.logprob_mode,
            logprob_sigma=self.logprob_sigma,
            decode_video=False,
        )
        if "ignore_first" in sig.parameters:
            call_kwargs["ignore_first"] = self.ignore_first
        if "ignore_last" in sig.parameters:
            call_kwargs["ignore_last"] = self.ignore_last
        if self.add_value_head:
            if "return_value_features" not in sig.parameters:
                raise RuntimeError(
                    "Motus PPO requires inference_step(return_value_features=...). "
                    "Update third_party/Motus/models/motus.py."
                )
            call_kwargs["return_value_features"] = True

        out = self._policy.model.inference_step(**call_kwargs)
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "Motus inference_step(return_trace=True) must return "
                "(predicted_frames, predicted_actions, trace)."
            )
        _, predicted_actions, trace = out[:3]

        required_trace_keys = {
            "video_chains",
            "action_chains",
            "denoise_inds",
            "video_latent_t",
            "action_x_t",
            "action_x_next",
            "t_scaled",
            "dt",
            "old_action_velocity",
        }
        if self.add_value_head:
            required_trace_keys.add("value_features")
        missing_trace_keys = required_trace_keys.difference(trace)
        if missing_trace_keys:
            raise RuntimeError(
                "Motus inference trace missing OpenPI-style replay keys: "
                f"{sorted(missing_trace_keys)}"
            )

        actions = predicted_actions[:, : self.num_action_chunks, : self.action_dim]
        action_tensor = actions.detach().float().cpu().contiguous()

        compute_values = self.add_value_head
        prev_logprobs, value_features, _entropy = self.model.get_log_prob_value(
            video_chains=trace["video_chains"],
            action_chains=trace["action_chains"],
            denoise_inds=trace["denoise_inds"],
            state=state,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs_batched,
            logprob_mode=self.logprob_mode,
            logprob_sigma=self.logprob_sigma,
            num_action_chunks=self.num_action_chunks,
            action_dim=self.action_dim,
            compute_values=compute_values,
            return_value_features=compute_values,
        )
        prev_logprobs = prev_logprobs.detach().float().cpu().contiguous()
        prev_values = None
        if compute_values:
            # Match OpenPI rollout behavior: compute a value for each denoise
            # step during sampling, then average values over the denoise dimension.
            prev_values = self._compute_values_from_features(trace["value_features"]).detach().float().cpu().contiguous()

        forward_inputs = {
            # Required by EnvWorker when storing executed actions.
            "action": action_tensor.reshape(batch_size, -1).contiguous(),
            "model_action": predicted_actions.detach().float().cpu().reshape(batch_size, -1).contiguous(),

            # Motus πRL context. Keep every field tensor-only and batch-major.
            "motus_first_frame": first_frame.detach().float().cpu().contiguous(),
            "motus_state": state.detach().float().cpu().contiguous(),
            "motus_language_embeddings": language_embeddings.detach().float().cpu().contiguous(),

            "motus_vlm_input_ids": vlm_inputs_batched["input_ids"].detach().cpu().contiguous(),
            "motus_vlm_attention_mask": vlm_inputs_batched["attention_mask"].detach().cpu().contiguous(),
            # Store pixel_values with explicit B dimension so RLinf split can split by env.
            "motus_vlm_pixel_values": vlm_inputs_batched["_pixel_values_batched"].detach().cpu().contiguous(),
            "motus_vlm_image_grid_thw": vlm_inputs_batched["image_grid_thw"].detach().cpu().contiguous(),

            # OpenPI-style denoise replay. Store full action/video chains and
            # the selected denoise index; actor forward replays the same transition.
            "motus_action_chains": trace["action_chains"].detach().float().cpu().contiguous(),
            "motus_video_chains": trace["video_chains"].detach().float().cpu().contiguous(),
            "motus_denoise_inds": trace["denoise_inds"].detach().cpu().contiguous(),

            # OpenPI-style PPO rollout values use per-denoise-step value features.
            **({
                "motus_value_features": trace["value_features"].detach().float().cpu().contiguous(),
            } if self.add_value_head else {}),

            # Compatibility/debug selected-transition fields. Actor replay is based
            # on chains + denoise_inds, not these selected-only tensors.
            "motus_video_latent_t": trace["video_latent_t"].detach().float().cpu().contiguous(),
            "motus_action_x_t": trace["action_x_t"].detach().float().cpu().contiguous(),
            "motus_action_x_next": trace["action_x_next"].detach().float().cpu().contiguous(),
            "motus_t_scaled": trace["t_scaled"].detach().float().cpu().contiguous(),
            "motus_dt": trace["dt"].detach().float().cpu().contiguous(),
            "motus_old_action_velocity": trace["old_action_velocity"].detach().float().cpu().contiguous(),
        }

        return action_tensor, {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }

    def _get_motus_text_len(self, fallback: int) -> int:
        """Return official Motus/WAN text length, normally 512.

        The field location differs between wrappers, so query several places.
        """
        candidates = [
            getattr(getattr(self.model, "video_module", None), "text_len", None),
            getattr(getattr(getattr(self.model, "video_module", None), "config", None), "text_len", None),
            getattr(getattr(self.model, "video_model", None), "text_len", None),
            getattr(getattr(getattr(self.model, "video_model", None), "wan_model", None), "text_len", None),
            getattr(getattr(self.model, "config", None), "text_len", None),
        ]
        for x in candidates:
            if x is not None:
                # Motus/WAN TI2V text_len is expected to be 512. Do not allow
                # accidental shorter text_len to reproduce train/eval mismatch.
                return max(int(x), int(fallback), 512)

        # Motus/WAN TI2V text_len is expected to be 512.
        return max(int(fallback), 512)

    def _stack_t5_embeddings(self, t5_list: list[torch.Tensor]) -> torch.Tensor:
        """Pad/stack per-sample T5 embeddings into [B, text_len, D].

        Official Motus/WAN inference pads T5 context to fixed text_len, normally
        512. Train-mode rollout must match that context length; padding only to
        batch max length changes cross-attention behavior.
        """
        if not t5_list:
            raise ValueError("empty t5_list")

        max_len = max(int(x.shape[0]) for x in t5_list)
        target_len = self._get_motus_text_len(max_len)

        dim = int(t5_list[0].shape[-1])
        dtype = t5_list[0].dtype
        device = t5_list[0].device

        out = torch.zeros((len(t5_list), target_len, dim), dtype=dtype, device=device)

        for i, emb in enumerate(t5_list):
            emb = emb.to(device=device, dtype=dtype)
            n = min(int(emb.shape[0]), target_len)
            out[i, :n, :] = emb[:n]

        return out

    def _collate_vlm_inputs(
        self,
        vlm_inputs_list: list[dict[str, torch.Tensor]],
        *,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        """Collate list-form official VLM inputs into tensor-only batch.

        The returned dict can be passed to Motus. It also includes
        _pixel_values_batched [B,N,...] for RLinf-safe storage/splitting.
        """
        if not vlm_inputs_list:
            raise ValueError("empty vlm_inputs_list")

        ids_list = [x["input_ids"].squeeze(0).to(device) for x in vlm_inputs_list]
        mask_list = [x["attention_mask"].squeeze(0).to(device) for x in vlm_inputs_list]
        max_len = max(int(x.shape[0]) for x in ids_list)

        input_ids = torch.zeros(
            (len(ids_list), max_len),
            dtype=ids_list[0].dtype,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(mask_list), max_len),
            dtype=mask_list[0].dtype,
            device=device,
        )

        for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
            input_ids[i, : ids.shape[0]] = ids
            attention_mask[i, : mask.shape[0]] = mask

        pixel_values_list = [x["pixel_values"].to(device) for x in vlm_inputs_list]
        pixel_shape0 = tuple(pixel_values_list[0].shape)
        for i, pv in enumerate(pixel_values_list):
            if tuple(pv.shape) != pixel_shape0:
                raise ValueError(
                    "Motus VLM pixel_values have variable shape; add padding before training. "
                    f"sample0={pixel_shape0}, sample{i}={tuple(pv.shape)}"
                )
        pixel_values_batched = torch.stack(pixel_values_list, dim=0)  # [B,N,...]
        pixel_values = pixel_values_batched.reshape(-1, *pixel_values_batched.shape[2:])

        grid_list = []
        for x in vlm_inputs_list:
            grid = x.get("image_grid_thw", None)
            if grid is None:
                raise ValueError("Motus VLM image_grid_thw is required for training.")
            grid_list.append(grid.squeeze(0).to(device))
        image_grid_thw = torch.stack(grid_list, dim=0)  # [B,3]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "_pixel_values_batched": pixel_values_batched,
        }

    def _vlm_inputs_from_forward_inputs(
        self,
        forward_inputs: dict[str, torch.Tensor],
        *,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        pixel_values_batched = forward_inputs["motus_vlm_pixel_values"].to(device)
        pixel_values = pixel_values_batched.reshape(-1, *pixel_values_batched.shape[2:])
        return {
            "input_ids": forward_inputs["motus_vlm_input_ids"].to(device),
            "attention_mask": forward_inputs["motus_vlm_attention_mask"].to(device),
            "pixel_values": pixel_values,
            "image_grid_thw": forward_inputs["motus_vlm_image_grid_thw"].to(device),
        }

    def _select_trace_step(self, num_inference_steps: int) -> int:
        lo = 1 if self.ignore_first else 0
        hi = num_inference_steps - 1 - (1 if self.ignore_last else 0)
        if hi < lo:
            raise ValueError(
                f"Invalid Motus denoise index range [{lo}, {hi}] for "
                f"num_inference_steps={num_inference_steps}, "
                f"ignore_first={self.ignore_first}, ignore_last={self.ignore_last}."
            )

        if self.collect_denoise_step == "random":
            return int(torch.randint(lo, hi + 1, (1,)).item())

        trace_step = int(self.collect_denoise_step)
        if not (lo <= trace_step <= hi):
            raise ValueError(
                f"Motus collect_denoise_step={trace_step} is outside the allowed "
                f"range [{lo}, {hi}] after ignore_first/ignore_last filtering."
            )
        return trace_step

    def _motus_sample_actions_with_transition(self, *args, **kwargs):
        """Disabled adapter-local denoise loop.

        Motus RL rollout must use third_party.Motus.inference_step(return_trace=True)
        so train rollout, old logprob, and actor replay share the same model-side
        denoise implementation. Falling back here would silently reintroduce the
        selected-transition-only shrink path.
        """
        raise RuntimeError(
            "Adapter-local Motus denoise fallback is disabled. Update "
            "third_party/Motus/models/motus.py to provide inference_step(return_trace=True) "
            "with action/video chains and denoise_inds."
        )

    def _motus_velocity_step(
        self,
        *,
        video_latent: torch.Tensor,
        action_latent: torch.Tensor,
        state: torch.Tensor,
        t_scaled: torch.Tensor,
        language_embeddings: torch.Tensor,
        vlm_inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Strict wrapper for the shared Motus model-side denoise step."""
        model = self.model
        if not hasattr(model, "denoise_velocity_step"):
            raise RuntimeError(
                "Motus model must implement denoise_velocity_step(); adapter-local "
                "denoise reconstruction is disabled to keep rollout and actor replay aligned."
            )
        video_velocity, action_velocity, _ = model.denoise_velocity_step(
            video_latent=video_latent,
            action_latent=action_latent,
            state=state,
            t_scaled=t_scaled,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
        )
        return video_velocity, action_velocity

    def _flow_sde_mean_std(
        self,
        *,
        action_x_t: torch.Tensor,
        action_velocity: torch.Tensor,
        t_scaled: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """OpenPI-style flow_sde transition mean/std.

        This mirrors OpenPI's sample_mean_var_val flow_sde branch:
          x0_pred = x_t - v * t
          x1_pred = x_t + v * (1 - t)
          sigma_i = noise_level * sqrt(t / (1 - t))
          std = sqrt(delta) * sigma_i
        where delta = t - t_next = -dt for our decreasing timesteps.
        """
        if hasattr(self.model, "flow_sde_mean_std"):
            return self.model.flow_sde_mean_std(
                action_x_t=action_x_t,
                action_velocity=action_velocity,
                t_scaled=t_scaled,
                dt=dt,
                noise_level=float(self.logprob_sigma),
            )

        raise RuntimeError(
            "Motus model must implement flow_sde_mean_std(); adapter-local "
            "flow-SDE formula fallback is disabled to keep rollout and actor replay aligned."
        )

    def _transition_velocity_logprobs(
        self,
        *,
        action_velocity: torch.Tensor,
        action_x_t: torch.Tensor,
        action_x_next: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian logprob for a flow transition x_next = x_t + v_theta * dt."""
        sigma = float(self.logprob_sigma)
        if sigma <= 0:
            raise ValueError(f"logprob_sigma must be positive, got {sigma}")

        while dt.dim() < action_x_t.dim():
            dt = dt.unsqueeze(-1)

        if self.logprob_mode in {"flow_sde", "transition_sde"}:
            # t_scaled is needed for exact OpenPI-style flow_sde. If this function
            # is called without it, caller must use transition_velocity mode.
            raise RuntimeError(
                "_transition_velocity_logprobs no longer supports flow_sde without t_scaled. "
                "Use _transition_logprobs(..., t_scaled=...) instead."
            )
        else:
            target_velocity = (action_x_next - action_x_t) / dt
            diff = (action_velocity.float() - target_velocity.float()) / sigma
            log_norm = math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
            logprobs = -0.5 * diff.pow(2) - log_norm

        return logprobs[:, : self.num_action_chunks, : self.action_dim].float()

    def _transition_logprobs(
        self,
        *,
        action_velocity: torch.Tensor,
        action_x_t: torch.Tensor,
        action_x_next: torch.Tensor,
        t_scaled: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Logprob for selected denoise transition."""
        if self.logprob_mode in {"flow_sde", "transition_sde"}:
            mean_next, std = self._flow_sde_mean_std(
                action_x_t=action_x_t,
                action_velocity=action_velocity,
                t_scaled=t_scaled,
                dt=dt,
            )
            diff = (action_x_next.float() - mean_next.float()) / std.float()
            log_norm = torch.log(std.float()) + 0.5 * math.log(2.0 * math.pi)
            logprobs = -0.5 * diff.pow(2) - log_norm
            return logprobs[:, : self.num_action_chunks, : self.action_dim].float()

        return self._transition_velocity_logprobs(
            action_velocity=action_velocity,
            action_x_t=action_x_t,
            action_x_next=action_x_next,
            dt=dt,
        )

    def _prepare_one_for_vectorized(
        self, env_obs: dict[str, Any], idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], str]:
        """Use official Motus preprocessing for one sample, then return tensors for batch."""
        observation = self._build_official_observation(env_obs, idx)
        raw_instruction = self._get_instruction(env_obs, idx)
        full_instruction = self.scene_prefix + raw_instruction

        # Reuse official update_obs() so image resize/padding and state handling match
        # official Motus RoboTwin inference.
        self._reset_official_policy_transient_state()
        self._policy.update_obs(observation)

        frame = self._policy.obs_cache[-1].squeeze(0).detach()  # [C,H,W]
        state = self._policy.current_state.squeeze(0).detach()  # [14]

        t5_out = self._policy.t5_encoder([full_instruction], self.device)
        t5_emb = self._normalize_t5_output(t5_out)

        first_frame_pil = self._policy._tensor_to_pil_image(frame.detach().cpu())
        vlm_inputs = self._policy._preprocess_vlm_messages(full_instruction, first_frame_pil)

        return frame, state, t5_emb, vlm_inputs, raw_instruction

    @staticmethod
    def _normalize_t5_output(t5_out: Any) -> torch.Tensor:
        if torch.is_tensor(t5_out):
            if t5_out.dim() == 3 and t5_out.shape[0] == 1:
                return t5_out.squeeze(0)
            return t5_out
        if isinstance(t5_out, list):
            if len(t5_out) != 1:
                raise ValueError(f"Expected single T5 embedding, got list length {len(t5_out)}")
            emb = t5_out[0]
            if torch.is_tensor(emb) and emb.dim() == 3 and emb.shape[0] == 1:
                return emb.squeeze(0)
            return emb
        raise ValueError(f"Unexpected T5 encoder output type: {type(t5_out)!r}")

    def _ensure_motus_grid_batch_size(self, batch_size: int) -> None:
        """Keep Motus/WAN grid_sizes aligned with runtime batch size.

        Old TTS experiments showed Motus inference may use runtime B different
        from config.batch_size. WAN attention/unpatchify expects grid_sizes[0] == B.
        """
        model = self._policy.model
        grid_sizes = getattr(model, "grid_sizes", None)

        if (
            torch.is_tensor(grid_sizes)
            and grid_sizes.shape[0] == batch_size
            and getattr(model, "video_module", None) is not None
        ):
            return

        lat_t = 1 + model.config.num_video_frames // 4
        lat_h = model.config.video_height // 32
        lat_w = model.config.video_width // 32

        device = getattr(model, "device", self.device)
        new_grid = torch.tensor(
            [lat_t, lat_h, lat_w],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)

        model.grid_sizes = new_grid
        if hasattr(model, "video_module"):
            model.video_module.grid_sizes = new_grid

    def _reset_official_policy_transient_state(self) -> None:
        if hasattr(self._policy, "obs_cache"):
            self._policy.obs_cache.clear()
        if hasattr(self._policy, "action_cache"):
            self._policy.action_cache.clear()
        if hasattr(self._policy, "current_state"):
            self._policy.current_state = None
        if hasattr(self._policy, "current_state_norm"):
            self._policy.current_state_norm = None
        if hasattr(self._policy, "prev_action"):
            self._policy.prev_action = None

    def _validate_action_array(self, action: Any, sample_idx: int = 0) -> np.ndarray:
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim != 2:
            raise ValueError(
                f"Motus action for sample {sample_idx} must be [T,D], got shape {action_np.shape}"
            )
        if action_np.shape[0] != self.num_action_chunks:
            raise ValueError(
                f"Motus action chunk mismatch for sample {sample_idx}: "
                f"expected {self.num_action_chunks}, got {action_np.shape[0]}"
            )
        if action_np.shape[1] != self.action_dim:
            raise ValueError(
                f"Motus action dim mismatch for sample {sample_idx}: "
                f"expected {self.action_dim}, got {action_np.shape[1]}"
            )
        return action_np

    @staticmethod
    def _infer_batch_size(env_obs: dict[str, Any]) -> int:
        for key in ("states", "main_images", "task_descriptions"):
            value = env_obs.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
            if isinstance(value, np.ndarray):
                return int(value.shape[0])
            if isinstance(value, list):
                return len(value)
        raise ValueError(f"Cannot infer batch size from env_obs keys={list(env_obs.keys())}")

    @staticmethod
    def _take_batch(value: Any, idx: int) -> Any:
        if isinstance(value, torch.Tensor):
            return value[idx]
        if isinstance(value, np.ndarray):
            return value[idx]
        if isinstance(value, list):
            return value[idx]
        return value

    def _get_instruction(self, env_obs: dict[str, Any], idx: int) -> str:
        descriptions = env_obs.get("task_descriptions", None)
        if descriptions is None:
            return ""
        if isinstance(descriptions, str):
            return descriptions
        if isinstance(descriptions, list):
            return str(descriptions[idx])
        return str(self._take_batch(descriptions, idx))

    def _build_official_observation(self, env_obs: dict[str, Any], idx: int) -> dict[str, Any]:
        main = self._take_batch(env_obs["main_images"], idx)
        wrist = env_obs.get("wrist_images", None)
        state = self._take_batch(env_obs["states"], idx)

        head_img = self._to_hwc_uint8(main)
        left_img, right_img = self._extract_wrist_pair(wrist, idx)

        combined_image = self._combine_three_views(head_img, left_img, right_img)

        state_np = self._to_numpy(state).astype(np.float32).reshape(-1)
        if state_np.shape[0] != self.action_dim:
            raise ValueError(
                f"Motus state dim mismatch: expected {self.action_dim}, got {state_np.shape[0]}"
            )

        # Official deploy_policy.update_obs accepts {'image': ..., 'joint_action': {'vector': ...}}.
        return {
            "image": combined_image,
            "joint_action": {
                "vector": state_np,
            },
        }

    def _extract_wrist_pair(self, wrist_images: Any, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if wrist_images is None:
            raise ValueError(
                "Motus requires wrist_images. Ensure collect_wrist_camera=true in env config."
            )

        wrist_i = self._take_batch(wrist_images, idx)

        if isinstance(wrist_i, (list, tuple)):
            if len(wrist_i) == 0:
                raise ValueError("wrist_images list is empty")
            left = wrist_i[0]
            right = wrist_i[1] if len(wrist_i) > 1 else wrist_i[0]
            return self._to_hwc_uint8(left), self._to_hwc_uint8(right)

        arr = self._to_numpy(wrist_i)

        # Expected after batch indexing: [N, H, W, C] or [N, C, H, W].
        if arr.ndim == 4:
            left = arr[0]
            right = arr[1] if arr.shape[0] > 1 else arr[0]
            return self._to_hwc_uint8(left), self._to_hwc_uint8(right)

        # Single wrist image; duplicate as fallback.
        if arr.ndim == 3:
            img = self._to_hwc_uint8(arr)
            return img, img

        raise ValueError(f"Unsupported wrist_images shape after batch indexing: {arr.shape}")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)

    @classmethod
    def _to_hwc_uint8(cls, value: Any) -> np.ndarray:
        arr = cls._to_numpy(value)

        if arr.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {arr.shape}")

        # CHW -> HWC
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        if arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB image, got shape {arr.shape}")

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(arr)

    @staticmethod
    def _resize_hwc_uint8(img: np.ndarray, width: int, height: int) -> np.ndarray:
        pil = Image.fromarray(img, mode="RGB")
        pil = pil.resize((width, height), Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)

    @classmethod
    def _combine_three_views(
        cls, head_img: np.ndarray, left_img: np.ndarray, right_img: np.ndarray
    ) -> np.ndarray:
        # Match official Motus RoboTwin layout:
        #   head: 320x240
        #   left/right wrists: each 160x120
        #   final: [head; concat(left,right)] = 320x360 HWC
        head = cls._resize_hwc_uint8(head_img, width=320, height=240)
        left = cls._resize_hwc_uint8(left_img, width=160, height=120)
        right = cls._resize_hwc_uint8(right_img, width=160, height=120)
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)
        return np.ascontiguousarray(image)
