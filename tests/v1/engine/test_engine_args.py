# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vllm.config import VllmConfig
from vllm.config import MoEOffloadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import _xxhash


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching, (
        "V1 turns on prefix caching by default."
    )

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not vllm_config.cache_config.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching

    # default hash algorithm is "builtin"
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # set hash algorithm to sha256_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256_cbor"

    # set hash algorithm to sha256
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # an invalid hash algorithm raises an error
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(["--prefix-caching-hash-algo", "invalid"])


@pytest.mark.skipif(_xxhash is None, reason="xxhash not installed")
def test_prefix_caching_xxhash_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # set hash algorithm to xxhash (pickle)
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash"

    # set hash algorithm to xxhash_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash_cbor"


def test_defaults_with_usage_context():
    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config: VllmConfig = engine_args.create_engine_config(UsageContext.LLM_CLASS)

    from vllm.platforms import current_platform
    from vllm.utils.mem_constants import GiB_bytes

    device_memory = current_platform.get_device_total_memory()
    device_name = current_platform.get_device_name().lower()
    if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
        # For GPUs like H100, H200, and MI300x with >= 70GB memory
        default_llm_tokens = 16384
        default_server_tokens = 8192
        default_max_num_seqs = 1024
    else:
        default_llm_tokens = 8192
        default_server_tokens = 2048
        default_max_num_seqs = 256

    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_llm_tokens  # noqa: E501

    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_server_tokens  # noqa: E501


def test_moe_offload_cli_args():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--moe-cpu-offload",
            "--moe-gpu-limit",
            "0.5",
            "--moe-wave-min-tokens",
            "4096",
            "--moe-min-residency-steps",
            "7",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.moe_cpu_offload
    assert engine_args.moe_gpu_limit == 0.5
    assert engine_args.moe_wave_min_tokens == 4096
    assert engine_args.moe_min_residency_steps == 7


def test_moe_offload_config_requires_enablement():
    with pytest.raises(
        ValidationError,
        match="moe_gpu_limit requires moe_cpu_offload",
    ):
        MoEOffloadConfig(gpu_limit=0.5)


def test_moe_offload_batch_defaults_raise_wave_target():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        moe_cpu_offload=True,
        moe_wave_min_tokens=4096,
    )
    engine_args.max_num_batched_tokens = None
    engine_args.max_num_seqs = 32
    engine_args.enable_chunked_prefill = True

    engine_args._set_default_max_num_seqs_and_batched_tokens_args(
        usage_context=UsageContext.OPENAI_API_SERVER,
        model_config=SimpleNamespace(max_model_len=8192),
        parallel_config=SimpleNamespace(use_batched_dp_moe=False),
    )

    assert engine_args.max_num_batched_tokens == 4096


def test_moe_offload_preserves_explicit_batch_limit():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        moe_cpu_offload=True,
        moe_wave_min_tokens=4096,
        max_num_batched_tokens=1024,
        max_num_seqs=32,
        enable_chunked_prefill=True,
    )

    engine_args._set_default_max_num_seqs_and_batched_tokens_args(
        usage_context=UsageContext.OPENAI_API_SERVER,
        model_config=SimpleNamespace(max_model_len=8192),
        parallel_config=SimpleNamespace(use_batched_dp_moe=False),
    )

    assert engine_args.max_num_batched_tokens == 1024


def test_moe_offload_config_validates_active_budget_requires_enablement():
    with pytest.raises(
        ValidationError,
        match="moe_active_expert_budget requires moe_cpu_offload",
    ):
        MoEOffloadConfig(active_expert_budget=8)
