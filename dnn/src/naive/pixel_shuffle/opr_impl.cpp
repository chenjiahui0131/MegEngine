/**
 * \file dnn/src/naive/pixel_shuffle/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/naive/pixel_shuffle/opr_impl.h"
#include "src/naive/handle.h"

namespace megdnn
{
    namespace naive
    {
        void PixelShuffleImpl::exec(_megdnn_tensor_in input, _megdnn_tensor_out output, _megdnn_workspace workspace)
        {
            check_exec(input.layout, output.layout, workspace.size);

#define cb(DType)                                                          \
    if (input.layout.dtype.enumv() == DTypeTrait<DType>::enumv)            \
    {                                                                      \
        using ctype = typename DTypeTrait<DType>::ctype;                   \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(input, output)); \
        return;                                                            \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
            MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
            megdnn_assert_internal(0);
        }

        template <typename T>
        void PixelShuffleImpl::exec_internal(_megdnn_tensor_in input, _megdnn_tensor_out output)
        {
            auto upscale_factor = param().upscale_factor;
            auto square = upscale_factor * upscale_factor;
            auto ndim = input.layout.ndim;
            auto ptr_in = input.ptr<T>();
            auto ptr_out = output.ptr<T>();
            auto shape_in = input.layout.shape;
            auto shape_out = output.layout.shape;
            size_t idx_in[TensorShape::MAX_NDIM];
            size_t idx_out[TensorShape::MAX_NDIM];
            std::memset(idx_in, 0, sizeof(idx_in));
            std::memset(idx_out, 0, sizeof(idx_out));
            std::memset(ptr_out, 0, sizeof(T) * output.layout.total_nr_elems());
            do
            {
                rep(i, ndim - 3) idx_out[i] = idx_in[i];
                idx_out[ndim - 3] = idx_in[ndim - 3] / square;
                idx_out[ndim - 2] = idx_in[ndim - 2] * upscale_factor + (idx_in[ndim - 3] % square) / upscale_factor;
                idx_out[ndim - 1] = idx_in[ndim - 1] * upscale_factor + (idx_in[ndim - 3] % upscale_factor);
                auto offset_in = get_linear_addr(idx_in, shape_in, ndim);
                auto offset_out = get_linear_addr(idx_out, shape_out, ndim);
                ptr_out[offset_out] = ptr_in[offset_in];
            } while (get_next_addr(idx_in, shape_in, ndim));
        }

        void PixelShuffleBackwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                                            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
                                            _megdnn_workspace workspace)
        {
            check_exec(src.layout, dst.layout, diff.layout, grad.layout, workspace.size);

#define cb(DType)                                                                 \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv)                     \
    {                                                                             \
        using ctype = typename DTypeTrait<DType>::ctype;                          \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(src, dst, diff, grad)); \
        return;                                                                   \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
            MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
            megdnn_assert_internal(0);
        }

        template <typename T>
        void PixelShuffleBackwardImpl::exec_internal(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                                                     _megdnn_tensor_in diff, _megdnn_tensor_out grad)
        {
            auto upscale_factor = param().upscale_factor;
            auto square = upscale_factor * upscale_factor;
            auto ndim = src.layout.ndim;
            auto ptr_diff = diff.ptr<T>();
            auto ptr_grad = grad.ptr<T>();
            auto shape_diff = diff.layout.shape;
            auto shape_grad = grad.layout.shape;
            size_t idx_diff[TensorShape::MAX_NDIM];
            size_t idx_grad[TensorShape::MAX_NDIM];
            std::memset(idx_diff, 0, sizeof(idx_diff));
            std::memset(idx_grad, 0, sizeof(idx_grad));
            std::memset(ptr_grad, 0, sizeof(T) * grad.layout.total_nr_elems());
            do
            {
                rep(i, ndim - 3) idx_diff[i] = idx_grad[i];
                idx_diff[ndim - 3] = idx_grad[ndim - 3] / square;
                idx_diff[ndim - 2] = idx_grad[ndim - 2] * upscale_factor + (idx_grad[ndim - 3] % square) / upscale_factor;
                idx_diff[ndim - 1] = idx_grad[ndim - 1] * upscale_factor + (idx_grad[ndim - 3] % upscale_factor);
                auto offset_grad = get_linear_addr(idx_grad, shape_grad, ndim);
                auto offset_diff = get_linear_addr(idx_diff, shape_diff, ndim);
                ptr_grad[offset_grad] = ptr_diff[offset_diff];
            } while (get_next_addr(idx_grad, shape_grad, ndim));

            /*
            do
            {
                rep(i, ndim - 3) idx_grad[i] = idx_diff[i];
                idx_out[ndim - 3] = idx_in[ndim - 3] * square + (idx_in[ndim - 2] % upscale_factor) * upscale_factor + (idx_in[ndim - 1] % upscale_factor);
                idx_out[ndim - 2] = idx_in[ndim - 2] / upscale_factor;
                idx_out[ndim - 1] = idx_in[ndim - 1] / upscale_factor;

                auto offset_in = get_linear_addr(idx_in, shape_in, ndim);
                auto offset_out = get_linear_addr(idx_out, shape_out, ndim);
                ptr_out[offset_out] = ptr_in[offset_in];
            } while (get_next_addr(idx_in, shape_in, ndim));
            */
        }
    } // namespace naive
} // namespace megdnn