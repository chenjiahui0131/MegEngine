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

using namespace mgb;
using namespace opr;

/* ==================== PixelShuffleForward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(PixelShuffleForward);

PixelShuffleForward::PixelShuffleForward(VarNode *src,
                                         const Param &param,
                                         const OperatorNodeConfig &config)
    : Super{src->owner_graph(), config, "pixel_shuffle", {src}}
{
    init_megdnn_opr(*this, param);
    add_input({src});
    output(0)->dtype(src->dtype());
}

SymbolVar PixelShuffleForward::make(SymbolVar src, const Param &param,
                                    const OperatorNodeConfig &config)
{
    return src.insert_single_output_opr<PixelShuffleForward>(src.node(), param, config);
}

/* ==================== PixelShuffleBackward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(PixelShuffleBackward);

PixelShuffleBackward::PixelShuffleBackward(VarNode *src,
                                           const Param &param,
                                           const OperatorNodeConfig &config)
    : Super{src->owner_graph(), config, "pixel_shuffle_bwd", {src}}
{
    init_megdnn_opr(*this, param);
    add_input({src});
    output(0)->dtype(src->dtype());
}

SymbolVar PixelShuffleBackward::make(SymbolVar src, const Param &param,
                                     const OperatorNodeConfig &config)
{
    return src.insert_single_output_opr<PixelShuffleBackward>(src.node(), param, config);
}