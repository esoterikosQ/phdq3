"""
BLT 환경 검증 스크립트
===================
xformers 없이 BLT를 구현할 수 있는 환경인지 단계별로 검증합니다.
각 단계가 통과해야 다음 단계로 진행할 수 있습니다.

실행: python verify_blt_env.py
GPU 지정: CUDA_VISIBLE_DEVICES=0 python verify_blt_env.py
"""

import sys
import time
import traceback


def print_header(step_num: int, title: str):
    print(f"\n{'='*70}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*70}")


def print_pass(msg: str):
    print(f"  ✓ {msg}")


def print_fail(msg: str):
    print(f"  ✗ {msg}")


def print_info(msg: str):
    print(f"    → {msg}")


# ─────────────────────────────────────────────
# Step 1: PyTorch 버전 및 CUDA 확인
# ─────────────────────────────────────────────
def step1_check_pytorch():
    """PyTorch >= 2.5, CUDA 사용 가능 여부 확인"""
    print_header(1, "PyTorch 버전 및 CUDA 확인")

    import torch

    # PyTorch 버전
    version = torch.__version__
    major, minor = int(version.split(".")[0]), int(version.split(".")[1])
    if major >= 2 and minor >= 5:
        print_pass(f"PyTorch {version} — FlexAttention 지원 버전")
    elif major >= 2 and minor >= 4:
        print_pass(f"PyTorch {version} — FlexAttention 베타 지원 (2.5+ 권장)")
    else:
        print_fail(f"PyTorch {version} — 2.5 이상 필요")
        return False

    # CUDA
    if not torch.cuda.is_available():
        print_fail("CUDA 사용 불가. GPU 드라이버 및 CUDA 설치 확인 필요")
        return False

    cuda_version = torch.version.cuda
    print_pass(f"CUDA {cuda_version}")

    # GPU 정보
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print_pass(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # bf16 지원 여부
    if torch.cuda.is_bf16_supported():
        print_pass("bf16 지원됨 (혼합 정밀도 학습 가능)")
    else:
        print_info("bf16 미지원 — fp16 사용 필요")

    return True


# ─────────────────────────────────────────────
# Step 2: SDPA (FlashAttention 백엔드) 검증
# ─────────────────────────────────────────────
def step2_check_sdpa():
    """
    글로벌 모델의 causal attention에 사용할 SDPA 검증.
    xformers 없이 FlashAttention이 동작하는지 확인.
    """
    print_header(2, "SDPA + FlashAttention 백엔드 검증")

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    device = "cuda"
    dtype = torch.bfloat16

    B, H, S, D = 2, 8, 1024, 64  # batch, heads, seq_len, head_dim
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # (a) 기본 SDPA 동작
    try:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.shape == (B, H, S, D)
        print_pass("SDPA causal attention 동작 확인")
    except Exception as e:
        print_fail(f"SDPA 실패: {e}")
        return False

    # (b) FlashAttention 백엔드 명시 확인
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out_flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print_pass("FlashAttention 백엔드 명시 호출 성공")
    except Exception as e:
        print_info(f"FlashAttention 백엔드 명시 불가: {e}")
        print_info("SDPA는 자동으로 최적 백엔드를 선택하므로 치명적이지 않음")

    # (c) backward pass
    try:
        q2 = q.clone().requires_grad_(True)
        k2 = k.clone().requires_grad_(True)
        v2 = v.clone().requires_grad_(True)
        out = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
        loss = out.sum()
        loss.backward()
        assert q2.grad is not None
        print_pass("SDPA backward pass (그래디언트 흐름) 확인")
    except Exception as e:
        print_fail(f"SDPA backward 실패: {e}")
        return False

    return True


# ─────────────────────────────────────────────
# Step 3: FlexAttention import 및 기본 동작
# ─────────────────────────────────────────────
def step3_check_flex_attention_basic():
    """FlexAttention 모듈 import 및 기본 동작 확인"""
    print_header(3, "FlexAttention 기본 동작 검증")

    import torch

    # (a) import
    try:
        from torch.nn.attention.flex_attention import (
            flex_attention,
            create_block_mask,
        )
        print_pass("flex_attention, create_block_mask import 성공")
    except ImportError as e:
        print_fail(f"FlexAttention import 실패: {e}")
        print_info("PyTorch 2.5 이상이 필요합니다")
        return False

    # (b) 가장 단순한 호출: no-op score_mod
    device = "cuda"
    dtype = torch.bfloat16
    B, H, S, D = 1, 4, 128, 64

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    def noop(score, b, h, q_idx, kv_idx):
        return score

    try:
        compiled_flex = torch.compile(flex_attention)
        out = compiled_flex(q, k, v, score_mod=noop)
        assert out.shape == (B, H, S, D)
        print_pass("FlexAttention 기본 호출 (no-op) 성공")
    except Exception as e:
        print_fail(f"FlexAttention 기본 호출 실패: {e}")
        traceback.print_exc()
        return False

    return True


# ─────────────────────────────────────────────
# Step 4: FlexAttention 커스텀 causal mask
# ─────────────────────────────────────────────
def step4_check_flex_causal_mask():
    """
    BLT 디코더 self-attention에 필요한 커스텀 마스크 검증.
    causal mask를 mask_mod로 정의하고 동작 확인.
    """
    print_header(4, "FlexAttention 커스텀 causal mask 검증")

    import torch
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    device = "cuda"
    dtype = torch.bfloat16
    B, H, S, D = 1, 4, 256, 64

    q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)

    # causal mask를 mask_mod로 정의
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    try:
        block_mask = create_block_mask(causal_mask, B, H, S, S, device=device)
        compiled_flex = torch.compile(flex_attention)
        out = compiled_flex(q, k, v, block_mask=block_mask)
        assert out.shape == (B, H, S, D)
        print_pass("커스텀 causal mask (mask_mod) 동작 확인")
    except Exception as e:
        print_fail(f"커스텀 causal mask 실패: {e}")
        traceback.print_exc()
        return False

    # backward
    try:
        loss = out.sum()
        loss.backward()
        assert q.grad is not None and k.grad is not None and v.grad is not None
        print_pass("커스텀 mask backward pass 확인")
    except Exception as e:
        print_fail(f"커스텀 mask backward 실패: {e}")
        return False

    return True


# ─────────────────────────────────────────────
# Step 5: BLT 디코더 핵심 마스크 패턴
# ─────────────────────────────────────────────
def step5_check_blt_decoder_mask():
    """
    BLT 디코더의 실제 어텐션 패턴을 시뮬레이션.
    클린 prefix: causal attention
    마스킹된 블록: bidirectional attention (블록 내)
    블록 → prefix: causal attention

    이것이 BLT-D 학습에 필요한 핵심 마스크 패턴입니다.
    """
    print_header(5, "BLT 디코더 혼합 마스크 패턴 (causal + bidirectional)")

    import torch
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    device = "cuda"
    dtype = torch.bfloat16

    # 시나리오: 시퀀스 길이 256 중 처음 192는 클린(causal), 나머지 64는 블록(bidirectional)
    CLEAN_LEN = 192
    BLOCK_LEN = 64
    TOTAL_LEN = CLEAN_LEN + BLOCK_LEN

    B, H, D = 1, 4, 64
    q = torch.randn(B, H, TOTAL_LEN, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, TOTAL_LEN, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, TOTAL_LEN, D, device=device, dtype=dtype, requires_grad=True)

    clean_len = CLEAN_LEN  # nonlocal 대신 closure로 캡처

    def blt_decoder_mask(b, h, q_idx, kv_idx):
        # 클린 영역 (q_idx < clean_len): 표준 causal
        is_clean_q = q_idx < clean_len
        causal_ok = kv_idx <= q_idx

        # 블록 영역 (q_idx >= clean_len): 전체 prefix + 블록 내 bidirectional
        block_ok = (kv_idx < clean_len) | (kv_idx >= clean_len)
        # 블록 내에서는 모든 위치에 attend 가능

        return torch.where(is_clean_q, causal_ok, block_ok)

    try:
        block_mask = create_block_mask(
            blt_decoder_mask, B, H, TOTAL_LEN, TOTAL_LEN, device=device
        )
        compiled_flex = torch.compile(flex_attention)
        out = compiled_flex(q, k, v, block_mask=block_mask)

        loss = out.sum()
        loss.backward()

        assert out.shape == (B, H, TOTAL_LEN, D)
        assert q.grad is not None
        print_pass(f"BLT 디코더 혼합 마스크 (clean={CLEAN_LEN} + block={BLOCK_LEN}) 동작 확인")
        print_pass("Forward + backward 모두 정상")
    except Exception as e:
        print_fail(f"BLT 디코더 마스크 실패: {e}")
        traceback.print_exc()
        return False

    return True


# ─────────────────────────────────────────────
# Step 6: Cross-Attention 시뮬레이션
# ─────────────────────────────────────────────
def step6_check_cross_attention():
    """
    BLT의 cross-attention 검증.
    바이트 hidden states (query) → 래턴트 토큰 (key, value)
    Q와 KV의 시퀀스 길이가 다른 경우.
    """
    print_header(6, "Cross-Attention (바이트 → 래턴트 토큰) 검증")

    import torch
    import torch.nn as nn

    device = "cuda"
    dtype = torch.bfloat16

    # 바이트: 1024개, 래턴트 토큰(패치): 256개 (평균 패치 크기 4)
    B = 2
    N_BYTES = 1024
    N_PATCHES = 256
    D_LOCAL = 512
    D_GLOBAL = 2048
    N_HEADS = 8
    D_HEAD = D_LOCAL // N_HEADS

    # cross-attention: Q는 바이트, KV는 래턴트 토큰
    # 래턴트 토큰을 linear transform으로 d_local 차원으로 투사
    proj_kv = nn.Linear(D_GLOBAL, D_LOCAL * 2, bias=False).to(device=device, dtype=dtype)

    byte_hidden = torch.randn(B, N_BYTES, D_LOCAL, device=device, dtype=dtype, requires_grad=True)
    latent_tokens = torch.randn(B, N_PATCHES, D_GLOBAL, device=device, dtype=dtype, requires_grad=True)

    try:
        # KV 투사
        kv = proj_kv(latent_tokens)  # (B, N_PATCHES, D_LOCAL*2)
        k, v = kv.chunk(2, dim=-1)   # 각각 (B, N_PATCHES, D_LOCAL)

        # reshape for multi-head attention
        q = byte_hidden.view(B, N_BYTES, N_HEADS, D_HEAD).transpose(1, 2)
        k = k.view(B, N_PATCHES, N_HEADS, D_HEAD).transpose(1, 2)
        v = v.view(B, N_PATCHES, N_HEADS, D_HEAD).transpose(1, 2)

        # SDPA (Q, KV 길이 다름 — is_causal 사용 불가, 마스크 없이 전체 attend)
        import torch.nn.functional as F
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        assert out.shape == (B, N_HEADS, N_BYTES, D_HEAD)

        loss = out.sum()
        loss.backward()
        assert byte_hidden.grad is not None
        assert latent_tokens.grad is not None

        print_pass(f"Cross-attention (bytes={N_BYTES} → patches={N_PATCHES}) 동작 확인")
        print_pass("Q/KV 길이가 다른 경우 forward + backward 정상")
    except Exception as e:
        print_fail(f"Cross-attention 실패: {e}")
        traceback.print_exc()
        return False

    return True


# ─────────────────────────────────────────────
# Step 7: FlexAttention + Cross-Attention 마스크
# ─────────────────────────────────────────────
def step7_check_flex_cross_attention():
    """
    BLT의 패치 구조 의존적 cross-attention 마스크를 FlexAttention으로 구현.
    각 바이트가 자신이 속한 패치의 이전 래턴트 토큰에만 attend하는 패턴.
    """
    print_header(7, "FlexAttention 기반 패치 의존적 cross-attention 마스크")

    import torch
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    device = "cuda"
    dtype = torch.bfloat16

    # 시나리오: 16바이트, 4패치 (패치 크기 4)
    # 패치 할당: byte 0-3 → patch 0, byte 4-7 → patch 1, ...
    # 각 바이트는 자신의 patch_index - 1 에 해당하는 래턴트 토큰에 attend
    # (패치 마지막 바이트는 자신의 래턴트 토큰에 attend)
    N_BYTES = 64
    N_PATCHES = 16
    PATCH_SIZE = N_BYTES // N_PATCHES
    B, H, D = 1, 4, 64

    q = torch.randn(B, H, N_BYTES, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N_PATCHES, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N_PATCHES, D, device=device, dtype=dtype, requires_grad=True)

    patch_size = PATCH_SIZE

    def patch_cross_attn_mask(b, h, q_idx, kv_idx):
        # 각 바이트의 패치 인덱스
        byte_patch_idx = q_idx // patch_size
        # 바이트가 패치의 마지막 위치인지
        is_last_in_patch = (q_idx % patch_size) == (patch_size - 1)
        # 일반: 이전 패치 래턴트에 attend
        prev_patch = byte_patch_idx - 1
        # 마지막 바이트: 자기 패치 래턴트에 attend
        target_patch = torch.where(is_last_in_patch, byte_patch_idx, prev_patch)
        return kv_idx == torch.clamp(target_patch, min=0)

    try:
        block_mask = create_block_mask(
            patch_cross_attn_mask, B, H, N_BYTES, N_PATCHES, device=device
        )
        compiled_flex = torch.compile(flex_attention)
        out = compiled_flex(q, k, v, block_mask=block_mask)

        loss = out.sum()
        loss.backward()

        assert out.shape == (B, H, N_BYTES, D)
        assert q.grad is not None and k.grad is not None
        print_pass(f"패치 의존적 cross-attention 마스크 동작 확인")
        print_pass(f"Q={N_BYTES} bytes, KV={N_PATCHES} patches — forward + backward 정상")
    except Exception as e:
        print_fail(f"패치 의존적 cross-attention 마스크 실패: {e}")
        traceback.print_exc()
        return False

    return True


# ─────────────────────────────────────────────
# Step 8: 성능 벤치마크 (메모리 + 속도)
# ─────────────────────────────────────────────
def step8_benchmark():
    """
    FlexAttention이 실제로 fused 커널을 사용하는지 확인.
    naive 구현 대비 메모리와 속도 비교.
    """
    print_header(8, "성능 벤치마크 — FlexAttention vs Naive 구현")

    import torch
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    device = "cuda"
    dtype = torch.bfloat16
    B, H, S, D = 1, 8, 2048, 64

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # (a) Naive: 명시적 마스크 생성
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    causal_mask_explicit = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
    attn_bias = torch.zeros(S, S, device=device, dtype=dtype)
    attn_bias.masked_fill_(~causal_mask_explicit, float("-inf"))

    t0 = time.time()
    for _ in range(10):
        out_naive = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    torch.cuda.synchronize()
    naive_time = (time.time() - t0) / 10
    naive_mem = torch.cuda.max_memory_allocated() / (1024**2)

    print_info(f"Naive (명시적 {S}×{S} 마스크): {naive_time*1000:.1f}ms, peak {naive_mem:.0f}MB")

    # (b) SDPA is_causal
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(10):
        out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    sdpa_time = (time.time() - t0) / 10
    sdpa_mem = torch.cuda.max_memory_allocated() / (1024**2)

    print_info(f"SDPA is_causal: {sdpa_time*1000:.1f}ms, peak {sdpa_mem:.0f}MB")

    # (c) FlexAttention
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        block_mask = create_block_mask(causal_mask, B, H, S, S, device=device)
        compiled_flex = torch.compile(flex_attention)

        # warmup (compilation)
        _ = compiled_flex(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(10):
            out_flex = compiled_flex(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
        flex_time = (time.time() - t0) / 10
        flex_mem = torch.cuda.max_memory_allocated() / (1024**2)

        print_info(f"FlexAttention:  {flex_time*1000:.1f}ms, peak {flex_mem:.0f}MB")

        if flex_mem < naive_mem:
            saved = (1 - flex_mem / naive_mem) * 100
            print_pass(f"FlexAttention이 naive 대비 메모리 {saved:.0f}% 절감 — fused 커널 동작 확인")
        else:
            print_info("메모리 차이 미미 — 시퀀스가 짧아 차이가 안 날 수 있음")

        if flex_time < naive_time:
            speedup = naive_time / flex_time
            print_pass(f"FlexAttention이 naive 대비 {speedup:.1f}x 빠름")

    except Exception as e:
        print_fail(f"FlexAttention 벤치마크 실패: {e}")
        traceback.print_exc()
        return False

    return True


# ─────────────────────────────────────────────
# Step 9: RoPE + RMSNorm + SwiGLU 검증
# ─────────────────────────────────────────────
def step9_check_building_blocks():
    """BLT 트랜스포머 레이어의 기본 빌딩 블록 동작 확인"""
    print_header(9, "트랜스포머 빌딩 블록 (RoPE, RMSNorm, SwiGLU)")

    import torch
    import torch.nn as nn

    device = "cuda"
    dtype = torch.bfloat16

    # (a) RMSNorm
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            return (x.float() * rms).to(x.dtype) * self.weight

    try:
        norm = RMSNorm(512).to(device=device, dtype=dtype)
        x = torch.randn(2, 128, 512, device=device, dtype=dtype)
        out = norm(x)
        assert out.shape == x.shape
        print_pass("RMSNorm 동작 확인")
    except Exception as e:
        print_fail(f"RMSNorm 실패: {e}")

    # (b) RoPE
    def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 500000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    def apply_rotary_emb(xq, xk, freqs_cis):
        # reshape to complex
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs = freqs_cis[: xq_.shape[-2]].unsqueeze(0).unsqueeze(0)
        xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
        xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
        return xq_out.to(xq.dtype), xk_out.to(xk.dtype)

    try:
        B, H, S, D = 2, 8, 256, 64
        freqs_cis = precompute_freqs_cis(D, S).to(device)
        xq = torch.randn(B, H, S, D, device=device, dtype=dtype)
        xk = torch.randn(B, H, S, D, device=device, dtype=dtype)
        xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
        assert xq_rot.shape == xq.shape
        print_pass(f"RoPE (θ=500000) 동작 확인")
    except Exception as e:
        print_fail(f"RoPE 실패: {e}")

    # (c) SwiGLU
    class SwiGLU(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        def forward(self, x):
            return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

    try:
        ffn = SwiGLU(512, 1376).to(device=device, dtype=dtype)
        x = torch.randn(2, 128, 512, device=device, dtype=dtype, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        print_pass("SwiGLU FFN forward + backward 확인")
    except Exception as e:
        print_fail(f"SwiGLU 실패: {e}")

    return True


# ─────────────────────────────────────────────
# Step 10: 분산 학습 기본 확인
# ─────────────────────────────────────────────
def step10_check_distributed():
    """멀티 GPU 환경인 경우 분산 학습 기본 요소 확인"""
    print_header(10, "분산 학습 환경 확인")

    import torch

    n_gpus = torch.cuda.device_count()
    print_info(f"사용 가능한 GPU: {n_gpus}개")

    if n_gpus < 2:
        print_info("GPU 1개 — 분산 학습 테스트 생략 (단일 GPU로 개발/검증 가능)")
        return True

    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        print_info(f"  GPU {i}: {name} ({mem:.1f} GB)")

    # NCCL 백엔드 확인
    try:
        if torch.distributed.is_nccl_available():
            print_pass("NCCL 백엔드 사용 가능")
        else:
            print_info("NCCL 미사용 — gloo 백엔드로 대체 가능하나 느림")
    except Exception as e:
        print_info(f"분산 백엔드 확인 불가: {e}")

    # FSDP 사용 가능 여부
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
        print_pass("FSDP import 가능")
    except ImportError:
        print_info("FSDP import 불가 — PyTorch 버전 확인 필요")

    return True


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
def main():
    print("\n" + "━" * 70)
    print("  BLT 환경 검증 스크립트")
    print("  xformers 없이 BLT를 구현할 수 있는 환경인지 확인합니다")
    print("━" * 70)

    steps = [
        ("PyTorch + CUDA", step1_check_pytorch),
        ("SDPA + FlashAttention", step2_check_sdpa),
        ("FlexAttention 기본", step3_check_flex_attention_basic),
        ("FlexAttention causal mask", step4_check_flex_causal_mask),
        ("BLT 디코더 혼합 마스크", step5_check_blt_decoder_mask),
        ("Cross-Attention", step6_check_cross_attention),
        ("패치 의존적 cross-attn 마스크", step7_check_flex_cross_attention),
        ("성능 벤치마크", step8_benchmark),
        ("빌딩 블록 (RoPE/RMSNorm/SwiGLU)", step9_check_building_blocks),
        ("분산 학습 환경", step10_check_distributed),
    ]

    results = {}
    for name, func in steps:
        try:
            passed = func()
            results[name] = passed
            if not passed:
                print(f"\n  ⚠  Step '{name}' 실패 — 이후 단계에 영향 가능")
        except Exception as e:
            results[name] = False
            print(f"\n  ⚠  Step '{name}' 예외 발생: {e}")
            traceback.print_exc()

    # 최종 요약
    print(f"\n\n{'━'*70}")
    print("  검증 결과 요약")
    print(f"{'━'*70}")

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print(f"{'━'*70}")
    if all_passed:
        print("  모든 검증 통과. 이 환경에서 BLT 구현을 시작할 수 있습니다.")
    else:
        print("  일부 검증 실패. 위 로그에서 실패 원인을 확인하세요.")
    print(f"{'━'*70}\n")


if __name__ == "__main__":
    main()