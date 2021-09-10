/**
 * \file dnn/src/common/pixel_shuffle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn
{
    void PixelShuffleBase::deduce_layout_fwd(const TensorLayout &input, TensorLayout &output)
    {
        megdnn_assert(input.ndim >= 3);
        auto ndim = input.ndim;
        auto shape = input.shape;
        auto upscale_factor = param().upscale_factor;
        auto square = upscale_factor * upscale_factor;
        megdnn_assert(shape[ndim - 3] % square == 0);
        SmallVector<size_t> shape_new(ndim);
        rep(i, ndim - 3) shape_new[i] = shape[i];
        shape_new[ndim - 3] = shape[ndim - 3] / square;
        shape_new[ndim - 2] = shape[ndim - 2] * upscale_factor;
        shape_new[ndim - 1] = shape[ndim - 1] * upscale_factor;
        output = TensorLayout(TensorShape(shape_new), input.dtype);
    }

    void PixelShuffleBase::check_layout_fwd(const TensorLayout &input, const TensorLayout &output)
    {
        TensorLayout output_expected;
        megdnn_assert_eq_dtype(input, output);
        deduce_layout_fwd(input, output_expected);
        megdnn_assert_eq_shape(output_expected, output);
    }

    void PixelShuffle::check_exec(const TensorLayout &input, const TensorLayout &output, size_t workspace_in_bytes)
    {
        TensorLayout output_expected;
        megdnn_assert_eq_dtype(input, output);
        deduce_layout(input, output_expected);
        megdnn_assert_eq_layout(output_expected, output);

        auto required_workspace_in_bytes = get_workspace_in_bytes(input, output);
        megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    }

    void PixelShuffle::deduce_layout(const TensorLayout &input, TensorLayout &output)
    {
        deduce_layout_fwd(input, output);
    }

    void PixelShuffleBackward::check_exec(const TensorLayout &src,
                                          const TensorLayout &dst, const TensorLayout &diff, const TensorLayout &grad, size_t workspace_in_bytes)
    {
        check_layout_fwd(src, dst);
        megdnn_assert_eq_layout(src, grad);
        megdnn_assert_eq_layout(dst, diff);
        auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, dst, diff, grad);
        megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    }

} // namespace megdnn