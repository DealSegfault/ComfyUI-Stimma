"""Workflow-scoped SageAttention acceleration for MiniMax H3."""

import logging

import torch
from comfy.patcher_extension import CallbacksMP


logger = logging.getLogger(__name__)


class StimmaMiniMaxH3SageAttention:
    """Patch H3's imported attention aliases for one model execution.

    MiniMax imports ``optimized_attention`` by value, so changing Comfy's
    module-level function after startup does not affect H3. This node patches
    only MiniMax's model and VAE aliases during the model lifecycle and restores
    them during cleanup. It intentionally uses the conservative FP8 accumulator;
    the FP16 CUDA kernel can silently return black H3 videos on Blackwell.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Stimma/Optimization"
    EXPERIMENTAL = True

    def patch(self, model):
        model_clone = model.clone()
        originals = {}

        @torch.compiler.disable()
        def enable(_model):
            from comfy.ldm.minimax import model as minimax_model
            from comfy.ldm.minimax import vae as minimax_vae

            originals["model"] = minimax_model.optimized_attention
            originals["vae"] = minimax_vae.optimized_attention
            minimax_model.optimized_attention = _attention_sage_fp8_safe
            minimax_vae.optimized_attention = _attention_sage_fp8_safe
            logger.info("Enabled workflow-scoped MiniMax H3 SageAttention FP8")

        @torch.compiler.disable()
        def disable(_model):
            from comfy.ldm.minimax import model as minimax_model
            from comfy.ldm.minimax import vae as minimax_vae

            if "model" in originals:
                minimax_model.optimized_attention = originals["model"]
            if "vae" in originals:
                minimax_vae.optimized_attention = originals["vae"]
            logger.info("Restored MiniMax H3 attention")

        model_clone.add_callback(CallbacksMP.ON_PRE_RUN, enable)
        model_clone.add_callback(CallbacksMP.ON_CLEANUP, disable)
        return (model_clone,)


@torch.compiler.disable()
def _attention_sage_fp8_safe(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    """Comfy attention adapter for Sage's accurate FP8 CUDA kernel."""
    from comfy.ldm.modules.attention import attention_pytorch
    from sageattention import sageattn_qk_int8_pv_fp8_cuda

    if mask is not None:
        return attention_pytorch(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    if skip_reshape:
        batch, _, _, dim_head = q.shape
        tensor_layout = "HND"
    else:
        batch, _, dim = q.shape
        dim_head = dim // heads
        q, k, v = (tensor.view(batch, -1, heads, dim_head) for tensor in (q, k, v))
        tensor_layout = "NHD"

    try:
        out = sageattn_qk_int8_pv_fp8_cuda(
            q,
            k,
            v,
            tensor_layout=tensor_layout,
            is_causal=False,
            sm_scale=kwargs.get("scale"),
            qk_quant_gran="per_warp",
            pv_accum_dtype="fp32+fp32",
        )
    except Exception as error:
        logger.warning("MiniMax H3 SageAttention failed; using PyTorch attention: %s", error)
        return attention_pytorch(
            q,
            k,
            v,
            heads,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    if tensor_layout == "HND":
        if not skip_output_reshape:
            out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
    elif skip_output_reshape:
        out = out.transpose(1, 2)
    else:
        out = out.reshape(batch, -1, heads * dim_head)
    return out


NODE_CLASS_MAPPINGS = {
    "StimmaMiniMaxH3SageAttention": StimmaMiniMaxH3SageAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StimmaMiniMaxH3SageAttention": "Stimma MiniMax H3 SageAttention",
}
