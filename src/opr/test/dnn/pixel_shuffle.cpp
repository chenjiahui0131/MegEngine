/**
 * \file src/opr/test/dnn/pixel_shuffle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./legacy_checker.h"
#include "megbrain/opr/dnn/pixel_shuffle.h"
#include "megbrain/utils/persistent_cache.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/basic_arith_wrapper.h"

using namespace std;
using namespace mgb;

namespace
{
    TEST(TestOprDNN, PixelShuffleForward)
    {
        auto graph = ComputingGraph::make();
        HostTensorGenerator<> gen{0, 1000};
        opr::PixelShuffle::Param param{3};
        auto host_a = gen({2, 18, 3, 3});
        auto a = opr::Host2DeviceCopy::make(*graph, host_a),
             b = opr::PixelShuffleForward::make(a, param);
        HostTensorND host_b;
        auto func = graph->compile({make_callback_copy(b, host_b)});
        func->execute();

        ASSERT_EQ(host_b.layout().shape[0], 2);
        ASSERT_EQ(host_b.layout().shape[1], 2);
        ASSERT_EQ(host_b.layout().shape[2], 9);
        ASSERT_EQ(host_b.layout().shape[3], 9);
    }

    TEST(TestOprDNN, PixelShuffleBackward)
    {
        auto graph = ComputingGraph::make();
        HostTensorGenerator<> gen{0, 1000};
        opr::PixelShuffle::Param param{3};
        auto host_a = gen({2, 18, 3, 3});
        auto host_b = gen({2, 2, 9, 9});
        auto host_c = gen({2, 2, 9, 9});
        auto a = opr::Host2DeviceCopy::make(*graph, host_a);
        auto b = opr::Host2DeviceCopy::make(*graph, host_b);
        auto c = opr::Host2DeviceCopy::make(*graph, host_c);
        auto d = opr::PixelShuffleBackward::make(a, b, c, param);
        HostTensorND host_d;
        auto func = graph->compile({make_callback_copy(d, host_d)});
        func->execute();

        ASSERT_EQ(host_d.layout().shape[0], 2);
        ASSERT_EQ(host_d.layout().shape[1], 18);
        ASSERT_EQ(host_d.layout().shape[2], 3);
        ASSERT_EQ(host_d.layout().shape[3], 3);
    }

    TEST(TestOprDNN, PixelShuffleForwardBackward)
    {
        auto graph = ComputingGraph::make();
        HostTensorGenerator<> gen{0, 1000};
        opr::PixelShuffle::Param param{3};
        auto host_a = gen({2, 18, 3, 3});
        auto a = opr::Host2DeviceCopy::make(*graph, host_a),
             b = opr::PixelShuffleForward::make(a, param);
        HostTensorND host_b;
        auto func = graph->compile({make_callback_copy(b, host_b)});
        func->execute();
        auto c = opr::PixelShuffleBackward::make(a, b, b, param);
        HostTensorND host_c;
        func = graph->compile({make_callback_copy(c, host_c)});
        func->execute();

        ASSERT_EQ(host_c.layout().shape[0], 2);
        ASSERT_EQ(host_c.layout().shape[1], 18);
        ASSERT_EQ(host_c.layout().shape[2], 3);
        ASSERT_EQ(host_c.layout().shape[3], 3);

        auto pa = host_a->ptr<float>();
        auto pc = host_c.ptr<float>();
        for (size_t i = 0; i < host_a->layout().total_nr_elems(); ++i)
        {
            ASSERT_EQ(pa[i], pc[i]);
        }
    }
}