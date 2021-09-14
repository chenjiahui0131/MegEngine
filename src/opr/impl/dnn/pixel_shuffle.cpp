/**
 * \file src/opr/impl/dnn/pixel_shuffle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megbrain/opr/dnn/pixel_shuffle.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/utility.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "megbrain/graph/grad_impl.h"

using namespace mgb;
using namespace opr;

/* ==================== PixelShuffleForward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(PixelShuffleForward);
MEGDNN_OPR_INIT1(PixelShuffleForward, "pixel_shuffle");

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(PixelShuffleForward)
{
    mgb_assert(wrt_idx == 0);
    SymbolVar grad = PixelShuffleBackward::make(
        opr.input(0), opr.output(0), out_grad[0], opr.param());
    return grad.node();
}
#endif

/* ==================== PixelShuffleBackward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(PixelShuffleBackward);
MEGDNN_OPR_INIT3(PixelShuffleBackward, "pixel_shuffle_grad");