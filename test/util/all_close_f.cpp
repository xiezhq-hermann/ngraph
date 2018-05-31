/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cmath>

#include "util/all_close_f.hpp"

using namespace std;
using namespace ngraph;

union FloatUnion {
    float f;
    uint32_t i;
};

bool test::close_f(float a, float b, int mantissa_bits, int tolerance_bits)
{
    FloatUnion a_fu{a};
    FloatUnion b_fu{b};
    uint32_t a_uint = a_fu.i;
    uint32_t b_uint = b_fu.i;

    // If negative: convert to two's complement
    // If positive: mask with sign bit
    uint32_t sign_mask = static_cast<uint32_t>(1U) << 31;
    a_uint = (sign_mask & a_uint) ? (~a_uint + 1) : (sign_mask | a_uint);
    b_uint = (sign_mask & b_uint) ? (~b_uint + 1) : (sign_mask | b_uint);

    uint32_t distance = (a_uint >= b_uint) ? (a_uint - b_uint) : (b_uint - a_uint);

    // e.g. for float with 24 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
    // tolerance_bit_shift = 32 -           (1 +  8 + (24 -     1         ) - 2)
    //                       float_length    sign exp  mantissa implicit 1    tolerance_bits
    uint32_t tolerance_bit_shift = 32 - (1 + 8 + (mantissa_bits - 1) - tolerance_bits);
    uint32_t tolerance = static_cast<uint32_t>(1U) << tolerance_bit_shift;

    // return distance <= tolerance;

    bool is_close = distance <= tolerance;

    if (!is_close)
    {
        float near_zero_bound = 0.0078125f;
        float upper_bound = 0.f + near_zero_bound;
        float lower_bound = 0.f - near_zero_bound;

        is_close =
            (a <= upper_bound) && (a >= lower_bound) && (b <= upper_bound) && (b >= lower_bound);
    }

    return is_close;
}

bool test::all_close_f(const vector<float>& a,
                       const vector<float>& b,
                       int mantissa_bits,
                       int tolerance_bits)
{
    bool rc = true;
    if (a.size() != b.size())
    {
        throw ngraph_error("a.size() != b.size() for all_close comparison.");
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        bool is_close_f = test::close_f(a[i], b[i], mantissa_bits, tolerance_bits);
        if (!is_close_f)
        {
            NGRAPH_INFO << a[i] << " !≈ " << b[i];
            rc = false;
        }
    }
    return rc;
}

bool test::all_close_f(const std::shared_ptr<runtime::TensorView>& a,
                       const std::shared_ptr<runtime::TensorView>& b,
                       int mantissa_bits,
                       int tolerance_bits)
{
    // Check that the layouts are compatible
    if (*a->get_tensor_view_layout() != *b->get_tensor_view_layout())
    {
        throw ngraph_error("Cannot compare tensors with different layouts");
    }
    if (a->get_shape() != b->get_shape())
    {
        return false;
    }

    return test::all_close_f(
        read_float_vector(a), read_float_vector(b), mantissa_bits, tolerance_bits);
}

bool test::all_close_f(const std::vector<std::shared_ptr<runtime::TensorView>>& as,
                       const std::vector<std::shared_ptr<runtime::TensorView>>& bs,
                       int mantissa_bits,
                       int tolerance_bits)
{
    if (as.size() != bs.size())
    {
        return false;
    }
    for (size_t i = 0; i < as.size(); ++i)
    {
        if (!test::all_close_f(as[i], bs[i], mantissa_bits, tolerance_bits))
        {
            return false;
        }
    }
    return true;
}
