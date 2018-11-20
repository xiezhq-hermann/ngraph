//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cstdint>
#include <memory>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/op/batch_norm.hpp"
#include "ngraph/frontend/onnx_import/utils/broadcasting.hpp"
#include "ngraph/frontend/onnx_import/utils/reshape.hpp"
#include "ngraph/frontend/onnx_import/utils/variadic.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector batch_norm(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto data = inputs.at(0);
                    const Shape& data_shape = data->get_shape();
                    auto scale = inputs.at(1);
                    auto bias = inputs.at(2);
                    std::shared_ptr<ngraph::Node> mean{nullptr};
                    std::shared_ptr<ngraph::Node> var{nullptr};

                    std::int64_t is_test{node.get_attribute_value<std::int64_t>("is_test", 1)};
                    std::int64_t spatial{node.get_attribute_value<std::int64_t>("spatial", 1)};
                    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};

                    // TODO: Implement learning mode support
                    // float momentum{node.get_attribute_value<float>("momentum", 0.9f)};
                    ASSERT_IS_SUPPORTED(node, is_test) << "only 'is_test' mode is supported.";

                    if (spatial == 0)
                    {
                        auto mean =
                            legacy_style_broadcast_for_binary_operation(data, inputs.at(3), 1)
                                .at(1);
                        auto variance =
                            legacy_style_broadcast_for_binary_operation(data, inputs.at(4), 1)
                                .at(1);

                        bias = legacy_style_broadcast_for_binary_operation(data, bias, 1).at(1);
                        scale = legacy_style_broadcast_for_binary_operation(data, scale, 1).at(1);

                        std::shared_ptr<ngraph::Node> epsilon_node = ngraph::op::Constant::create(
                            data->get_element_type(),
                            data_shape,
                            std::vector<double>(shape_size(data_shape), epsilon));

                        std::shared_ptr<ngraph::Node> one_node = ngraph::op::Constant::create(
                            data->get_element_type(),
                            data_shape,
                            std::vector<double>(shape_size(data_shape), 1));

                        return {(scale * ((data - mean) *
                                          (one_node / (std::make_shared<ngraph::op::Sqrt>(
                                                          variance + epsilon_node)))) +
                                 bias)};
                    }
                    else
                    {
                        if (inputs.size() >= 5)
                        {
                            mean = inputs.at(3);
                            var = inputs.at(4);
                            return {std::make_shared<ngraph::op::BatchNormInference>(
                                epsilon, scale, bias, data, mean, var)};
                        }

                        return {std::make_shared<ngraph::op::BatchNormTraining>(
                            epsilon, scale, bias, data)};
                    }
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
