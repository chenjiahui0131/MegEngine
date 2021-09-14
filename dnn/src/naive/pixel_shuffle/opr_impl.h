/**
 * \file dnn/src/naive/pixel_shuffle/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

namespace megdnn
{
    namespace naive
    {

        class PixelShuffleImpl : public PixelShuffle
        {
        public:
            using PixelShuffle::PixelShuffle;

            void exec(_megdnn_tensor_in input, _megdnn_tensor_out output, _megdnn_workspace workspace) override;

            size_t get_workspace_in_bytes(const TensorLayout &,
                                          const TensorLayout &) override
            {
                return 0;
            }

        private:
            template <typename T>
            void exec_internal(_megdnn_tensor_in input, _megdnn_tensor_out output);
        };

        class PixelShuffleBackwardImpl : public PixelShuffleBackward
        {
        public:
            using PixelShuffleBackward::PixelShuffleBackward;

            void exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                      _megdnn_tensor_in diff, _megdnn_tensor_out grad,
                      _megdnn_workspace workspace) override;

            size_t get_workspace_in_bytes(const TensorLayout &,
                                          const TensorLayout &,
                                          const TensorLayout &,
                                          const TensorLayout &) override
            {
                return 0;
            }

        private:
            template <typename T>
            void exec_internal(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                               _megdnn_tensor_in diff, _megdnn_tensor_out grad);
        };
    } //namespace naive
} // namespace megdnn