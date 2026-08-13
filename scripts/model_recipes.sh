#!/usr/bin/env bash
# Architecture presets used by run_real_moe.sh.
#
# This file is intentionally side-effect free until configure_model_recipe is
# called, which also makes each recipe cheap to validate without launching
# torchrun.

_configure_depth() {
  local full_num_layers="${1:?full layer count is required}"
  local dense_prefix_layers="${2:?dense prefix count is required}"
  MODEL_FULL_NUM_LAYERS="${full_num_layers}"
  MODEL_DENSE_PREFIX_LAYERS="${dense_prefix_layers}"
  MODEL_NUM_LAYERS="${NUM_LAYERS:-${MODEL_FULL_NUM_LAYERS}}"

  if ! [[ "${MODEL_NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid NUM_LAYERS=${MODEL_NUM_LAYERS} (expected a positive integer)" >&2
    return 1
  fi
  if (( MODEL_NUM_LAYERS > MODEL_FULL_NUM_LAYERS )); then
    echo "invalid NUM_LAYERS=${MODEL_NUM_LAYERS} (model maximum is ${MODEL_FULL_NUM_LAYERS})" >&2
    return 1
  fi
  if (( MODEL_NUM_LAYERS <= MODEL_DENSE_PREFIX_LAYERS )); then
    echo "invalid NUM_LAYERS=${MODEL_NUM_LAYERS} (must leave at least one MoE layer after the ${MODEL_DENSE_PREFIX_LAYERS}-layer dense prefix)" >&2
    return 1
  fi

  MODEL_MOE_LAYERS=$((MODEL_NUM_LAYERS - MODEL_DENSE_PREFIX_LAYERS))
  if (( MODEL_DENSE_PREFIX_LAYERS == 0 )); then
    MODEL_MOE_PATTERN="1"
  else
    MODEL_MOE_PATTERN="([0]*${MODEL_DENSE_PREFIX_LAYERS}+[1]*${MODEL_MOE_LAYERS})"
  fi
}

configure_model_recipe() {
  local model="${1:?model name is required}"
  MODEL_ARGS=()
  MOE_ARGS=()
  MODEL_DEFAULT_ROUTER_BALANCING="aux_loss"

  case "${model}" in
    mixtral8x7b)
      _configure_depth 32 0 || return
      MODEL_ARGS=(
        --use-mcore-models --disable-bias-linear --untie-embeddings-and-output-weights
        --seq-length "${SEQ_LEN:-4096}" --max-position-embeddings 32768
        --num-layers "${MODEL_NUM_LAYERS}" --hidden-size 4096 --ffn-hidden-size 14336
        --num-attention-heads 32 --group-query-attention --num-query-groups 8
        --normalization RMSNorm --position-embedding-type rope --rotary-base 1000000
        --swiglu --no-masked-softmax-fusion --no-position-embedding
        --attention-dropout 0.0 --hidden-dropout 0.0
      )
      MOE_ARGS=(
        --num-experts 8 --moe-router-topk 2
        --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-2
        --moe-token-dispatcher-type alltoall --moe-layer-freq "${MODEL_MOE_PATTERN}"
      )
      ;;

    qwen3_30b_a3b)
      _configure_depth 48 0 || return
      # Megatron Bridge Qwen checkpoints store this RMSNorm as
      # `pre_mlp_layernorm.*`; the local sequential spec otherwise aliases it
      # to the fused-linear key and leaves the norm unloaded.
      export EPLB_SEPARATE_MLP_NORM_CKPT="${EPLB_SEPARATE_MLP_NORM_CKPT:-1}"
      MODEL_ARGS=(
        --use-mcore-models --disable-bias-linear --untie-embeddings-and-output-weights
        --seq-length "${SEQ_LEN:-8192}" --max-position-embeddings 8192
        --num-layers "${MODEL_NUM_LAYERS}" --hidden-size 2048 --ffn-hidden-size 6144
        --num-attention-heads 32 --kv-channels 128
        --group-query-attention --num-query-groups 4 --qk-layernorm
        --normalization RMSNorm --norm-epsilon 1e-6
        --position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0
        --swiglu --no-masked-softmax-fusion --attention-softmax-in-fp32
        --attention-dropout 0.0 --hidden-dropout 0.0
        --padded-vocab-size 151936 --make-vocab-size-divisible-by 128
      )
      MOE_ARGS=(
        --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768
        --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3
        --moe-token-dispatcher-type alltoall --moe-layer-freq "${MODEL_MOE_PATTERN}"
      )
      ;;

    deepseek_v2_160e)
      # DeepSeek-V2: retain the production widths, MLA, routing, and two
      # shared experts. NUM_LAYERS can truncate the depth while preserving the
      # official one-layer dense prefix.
      _configure_depth 60 1 || return
      MODEL_DEFAULT_ROUTER_BALANCING="seq_aux_loss"
      MODEL_ARGS=(
        --use-mcore-models --disable-bias-linear --untie-embeddings-and-output-weights
        --seq-length "${SEQ_LEN:-4096}" --max-position-embeddings 163840
        --num-layers "${MODEL_NUM_LAYERS}" --hidden-size 5120 --ffn-hidden-size 12288
        --num-attention-heads 128 --kv-channels 128
        --multi-latent-attention --q-lora-rank 1536 --kv-lora-rank 512
        --qk-head-dim 128 --qk-pos-emb-head-dim 64 --v-head-dim 128
        --qk-layernorm --normalization RMSNorm --norm-epsilon 1e-6
        --position-embedding-type rope --rotary-base 10000
        --rotary-scaling-factor 40 --mscale 0.707 --mscale-all-dim 0.707
        --swiglu --attention-dropout 0.0 --hidden-dropout 0.0
        --init-method-std 0.006
        --padded-vocab-size 102400 --make-vocab-size-divisible-by 3200
      )
      MOE_ARGS=(
        --num-experts 160 --moe-router-topk 6 --moe-ffn-hidden-size 1536
        --moe-shared-expert-intermediate-size 3072
        --moe-layer-freq "${MODEL_MOE_PATTERN}"
        --moe-router-num-groups 8 --moe-router-group-topk 3
        --moe-router-pre-softmax --moe-router-score-function softmax
        --moe-router-topk-scaling-factor 16.0 --moe-router-dtype fp32
        --moe-router-load-balancing-type seq_aux_loss --moe-aux-loss-coeff 1e-3
        --moe-token-dispatcher-type alltoall
      )
      ;;

    glm45_air)
      # GLM-4.5-Air: retain GQA, 128 routed experts, and one serial shared
      # expert. NUM_LAYERS can truncate the depth while preserving the official
      # one-layer dense prefix. Shared-expert overlap stays disabled so
      # off/observe/apply use the same execution schedule.
      _configure_depth 46 1 || return
      MODEL_DEFAULT_ROUTER_BALANCING="seq_aux_loss"
      MODEL_ARGS=(
        --use-mcore-models --disable-bias-linear --add-qkv-bias
        --untie-embeddings-and-output-weights
        --seq-length "${SEQ_LEN:-4096}" --max-position-embeddings 131072
        --num-layers "${MODEL_NUM_LAYERS}" --hidden-size 4096 --ffn-hidden-size 10944
        --num-attention-heads 96 --kv-channels 128
        --group-query-attention --num-query-groups 8
        --normalization RMSNorm --norm-epsilon 1e-5
        --position-embedding-type rope --rotary-base 1000000 --rotary-percent 0.5
        --swiglu --attention-dropout 0.0 --hidden-dropout 0.0
        --init-method-std 0.02
        --padded-vocab-size 151552 --make-vocab-size-divisible-by 128
      )
      MOE_ARGS=(
        --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 1408
        --moe-shared-expert-intermediate-size 1408
        --moe-layer-freq "${MODEL_MOE_PATTERN}"
        --moe-router-score-function sigmoid --moe-router-enable-expert-bias
        --moe-router-bias-update-rate 0.0 --moe-router-topk-scaling-factor 1.0
        --moe-router-dtype fp32
        --moe-router-load-balancing-type seq_aux_loss --moe-aux-loss-coeff 1e-3
        --moe-token-dispatcher-type alltoall
      )
      ;;

    *)
      echo "unknown MODEL=${model} (expected qwen3_30b_a3b | mixtral8x7b | deepseek_v2_160e | glm45_air)" >&2
      return 1
      ;;
  esac
}
