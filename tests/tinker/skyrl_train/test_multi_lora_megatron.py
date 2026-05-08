"""End-to-end multi-LoRA tests against a Tinker server backed by SkyRL-Train Megatron.

GPU-gated: skipped automatically when no CUDA device is visible to the test
process. The server starts a real Megatron policy worker, which means tests
in this module need at least one GPU and the ``tinker`` + ``megatron`` extras
installed.

Coverage:
  - test_two_adapters_train_independently: A's optimizer state survives a
    swap-out, B-training, swap-in cycle (loss continues to improve on A).
  - test_rank_mismatch_rejected: a second create_model with different rank
    is hard-rejected at the controller-level signature gate.
  - test_sample_with_two_adapters_errors: v1 single-tenant guard on
    save_sampler_checkpoint fires when more than one LoRA is registered.
  - test_per_adapter_step_isolation: A's and B's pre-update losses are
    bit-exact when they were both pristine + saw identical data.
  - test_delete_then_train_remaining: deleting one of two adapters does
    not tear down the Ray runtime; the other adapter continues to train.

Run with:
  uv run --extra tinker --extra megatron --with pytest --with pytest-timeout \\
    pytest -s tests/tinker/skyrl_train/test_multi_lora_megatron.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager

import pytest

cuda_available = False
try:  # pragma: no cover - import guard
    import torch

    cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
except Exception:
    cuda_available = False

pytestmark = pytest.mark.skipif(not cuda_available, reason="multi-LoRA Megatron tests require at least one CUDA GPU")

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402

from tests.tinker.conftest import wait_for_condition  # noqa: E402

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINKER_API_KEY = "tml-dummy"
TEST_PORT = 8011

# Tiny config: 1 GPU, no TP/PP, single DP rank.
# With a tiny model + LoRA rank 8, this fits comfortably in
# any modern GPU.
BACKEND_CONFIG = {
    "strategy": "megatron",
    "trainer.placement.policy_num_gpus_per_node": 1,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": False,
    "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
}


@contextmanager
def _api_server(port: int, backend_config: dict | None = None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        db_path = os.path.join(tmp_dir, "server.db")
        cfg = dict(backend_config or BACKEND_CONFIG)
        cmd = [
            "uv",
            "run",
            "--extra",
            "tinker",
            "--extra",
            "megatron",
            "-m",
            "skyrl.tinker.api",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--base-model",
            BASE_MODEL,
            "--backend",
            "megatron",
            "--backend-config",
            json.dumps(cfg),
            "--database-url",
            f"sqlite:///{db_path}",
        ]
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            try:
                # Wait for server to come up
                ok = wait_for_condition(
                    lambda: _server_is_up(port),
                    timeout_sec=120,
                    poll_interval_sec=2,
                )
                if not ok:
                    with open(log_path) as f:
                        print(f"=== Server failed to start ===\n{f.read()}")
                    pytest.fail("Tinker API server did not come up in time")
                yield proc
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()


def _server_is_up(port: int) -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"http://0.0.0.0:{port}/api/v1/healthz", timeout=2).read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
        return False


def _make_datum(tokenizer, prompt: str, completion: str):
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f"{completion}\n\n", add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights[1:] + [1.0]},
    )


@pytest.fixture(scope="module")
def server():
    with _api_server(TEST_PORT) as proc:
        yield proc


@pytest.fixture
def service_client(server):
    return tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_PORT}/", api_key=TINKER_API_KEY)


def test_two_adapters_train_independently(service_client):
    """Two LoRA adapters share the same base model; training one must not
    contaminate the other's weights.

    We don't push weights to vLLM here because save_weights_for_sampler is
    deliberately gated to single-adapter in v1 (see
    test_sample_with_two_adapters_errors). Isolation is verified by
    asserting A's loss continues to improve after we've swapped to B and
    back — that's only possible if A's optimizer state survived the
    swap-out + B-training + swap-in cycle.
    """
    client_a = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    client_b = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = client_a.get_tokenizer()

    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]

    # Train A twice (priming + one real step)
    for _ in range(2):
        client_a.forward_backward(data, "cross_entropy").result()
        client_a.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()

    # Train B once with a different LR — this swaps the live adapter to B.
    client_b.forward_backward(data, "cross_entropy").result()
    client_b.optim_step(tinker_types.AdamParams(learning_rate=1e-4)).result()

    # Switch back to A. If A's optimizer/grad state was wiped by the swap,
    # the next step won't produce a sane gradient direction and loss won't
    # improve. Single-step convergence on a fixed micro-batch is reliable
    # for a tiny model + nontrivial LR.
    pre_loss = client_a.forward_backward(data, "cross_entropy").result()
    client_a.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()
    post_loss = client_a.forward_backward(data, "cross_entropy").result()
    pre = sum(sum(o["elementwise_loss"].data) for o in pre_loss.loss_fn_outputs)
    post = sum(sum(o["elementwise_loss"].data) for o in post_loss.loss_fn_outputs)
    assert post <= pre + 1e-3, (
        f"A's loss did not improve after a step (pre={pre}, post={post}); "
        "looks like A's optimizer state was wiped by the swap."
    )


def test_rank_mismatch_rejected(service_client):
    service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    with pytest.raises(Exception) as exc:
        service_client.create_lora_training_client(base_model=BASE_MODEL, rank=16)
    assert "signature mismatch" in str(exc.value).lower() or "rank" in str(exc.value).lower()


def test_sample_with_two_adapters_errors(service_client):
    a = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    with pytest.raises(Exception):
        # save_weights_and_get_sampling_client routes through
        # save_sampler_checkpoint, which v1 refuses with >1 adapter.
        a.save_weights_and_get_sampling_client(name="should_fail")


def test_per_adapter_step_isolation(service_client):
    """Two fresh adapters, identical pristine state, identical data. We
    interleave their fwd_bwd + optim_step calls (A.step0, B.step0, A.step1,
    B.step1) and assert that A's pre-update loss == B's pre-update loss
    bit-exact at every step. Both adapters were pristine when their first
    step ran, and both received the same parameters after their respective
    updates, so their losses must match — unless a step counter, scheduler
    position, or other Adam-bias-correction state leaks across adapters via
    shared optimizer state.
    """
    client_a = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    client_b = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = client_a.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]

    def fb_step(c):
        out = c.forward_backward(data, "cross_entropy").result()
        loss = sum(sum(o["elementwise_loss"].data) for o in out.loss_fn_outputs)
        c.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()
        return loss

    # ALT pattern: A.step0, B.step0, A.step1, B.step1
    a0 = fb_step(client_a)
    b0 = fb_step(client_b)
    a1 = fb_step(client_a)
    b1 = fb_step(client_b)
    print(
        f"\n[step_isolation] step 0: A={a0!r} B={b0!r} |Δ|={abs(a0 - b0):.6e}\n"
        f"[step_isolation] step 1: A={a1!r} B={b1!r} |Δ|={abs(a1 - b1):.6e}"
    )

    # Step 0: both adapters were pristine + saw identical data → bit-exact.
    assert a0 == b0, f"step 0 loss differs: A={a0!r} B={b0!r} (Δ={abs(a0 - b0):.6e})"

    # Step 1: both adapters had exactly one optim_step from pristine on
    # identical data. With AdapterStore correctly snapshotting both per-
    # param state and per-param-group state (TE FusedAdam tracks the
    # bias-correction step counter at the group level), both updates use
    # t=2 and the post-update parameters are bit-identical.
    assert a1 == b1, (
        f"step 1 loss diverges between adapters: A={a1!r} B={b1!r} (|Δ|={abs(a1 - b1):.6e}). "
        f"Symmetric prediction of a shared global step counter (TE FusedAdam's "
        f"`param_groups[g]['step']`) advancing on every optim_step instead of being "
        f"held per-adapter."
    )


def test_delete_then_train_remaining(service_client):
    a = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    b = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = a.get_tokenizer()
    data = [_make_datum(tok, "Q?", " a")]

    # Delete A via the unload_model endpoint (Tinker exposes this as the
    # public deletion path).
    async def _unload(model_id: str):
        async with tinker._client.AsyncTinker(  # type: ignore[attr-defined]
            api_key=TINKER_API_KEY, base_url=f"http://0.0.0.0:{TEST_PORT}/"
        ) as client:
            future = await client.models.unload(request=tinker_types.UnloadModelRequest(model_id=model_id))
            while True:
                result = await client.futures.retrieve(
                    request=tinker_types.FutureRetrieveRequest(request_id=future.request_id)
                )
                if isinstance(result, tinker_types.UnloadModelResponse):
                    return result
                await asyncio.sleep(0.1)

    asyncio.run(_unload(a.model_id))

    # B should still train successfully — backend should NOT have done a
    # ray.shutdown when only A was deleted.
    b.forward_backward(data, "cross_entropy").result()
    b.optim_step(tinker_types.AdamParams(learning_rate=1e-3)).result()
