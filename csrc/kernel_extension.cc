#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(lyra_w4afp8, m) {
  /*
   * Machete (Dense) Optimized Mixed Precision GEMM for Hopper.
   */
  m.def(
      "machete_supported_schedules("
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? maybe_group_scales_type,"
      "   ScalarType? maybe_group_zeros_type,"
      "   ScalarType? maybe_channel_scales_type,"
      "   ScalarType? maybe_token_scales_type,"
      "   ScalarType? maybe_out_type"
      ") -> str[]");
  m.impl("machete_supported_schedules", &machete::supported_schedules);

  m.def(
      "machete_mm("
      "   Tensor A,"
      "   Tensor B,"
      "   Tensor D,"
      "   int b_type,"
      "   ScalarType? out_type,"
      "   Tensor? group_scales,"
      "   Tensor? group_zeros,"
      "   int?    group_size,"
      "   Tensor? channel_scales,"
      "   Tensor? token_scales,"
      "   str?    schedule,"
      "   Tensor? group_layout,"
      "   Tensor? valid_len,"
      "   int?    group_stride"
      ") -> Tensor");
  m.impl("machete_mm", torch::kCUDA, &machete::mm);

  m.def(
      "machete_prepack_B("
      "   Tensor B,"
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? group_scales_type"
      ") -> Tensor");
  m.impl("machete_prepack_B", torch::kCUDA, &machete::prepack_B);
  // conditionally compiled so impl registration is in source file

  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! token_cnts_buffer, Tensor! cumsum_buffer, bool "
      "pad_sorted_token_ids) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

}

REGISTER_EXTENSION(common_ops)