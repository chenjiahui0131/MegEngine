/**
 * \file src/opr/include/megbrain/opr/dnn/pixel_shuffle.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"

namespace mgb{
namespace opr{

MGB_DEFINE_OPR_CLASS(PixelShuffleForward,
                     intl::MegDNNOprWrapperFwd<megdnn::PixelShuffleForward>) //{
public:
    PixelShuffleForward(VarNode *src, const Param &param,
                        const OperatorNodeConfig &config);
    static SymbolVar make(SymbolVar src, const Param &param,
                          const OperatorNodeConfig &config = {});
};
using PixelShuffle = PixelShuffleForward;

MGB_DEFINE_OPR_CLASS(PixelShuffleBackward,
                     intl::MegDNNOprWrapperFwd<megdnn::PixelShuffleBackward>) //{
public:
    PixelShuffleBackward(VarNode *src, VarNode *dst, VarNode *diff, const Param &param,
                         const OperatorNodeConfig &config);
    static SymbolVar make(SymbolVar src, SymbolVar dst, SymbolVar diff, const Param &param,
                          const OperatorNodeConfig &config = {});
};

} // namespace opr
} // namespace mgb
