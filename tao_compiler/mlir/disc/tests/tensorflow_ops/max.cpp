/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mlir/disc/tests/mlir_feature_test.h"
#include "mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"

namespace mlir_test {

const std::string c_ft_path = "mlir/disc/tests/tensorflow_ops/data/";

// // static shape 3D column reduction test case
// TEST(TFMaxOpTest, ColReduceStaticShape3DF32) {
//   setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
//   setenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", "true", 1);
//   setenv("DISC_ENABLE_STITCH", "true", 1);
//   // std::vector<float> input_val;
//   // for (int64_t i = 0; i < 110; i++) {
//   //   input_val.push_back(141701.0);
//   // }
//   // for (int64_t i = 0; i < 110 * 100 * 13; i++) {
//   //   input_val.push_back(1.0);
//   // }
//   EXPECT_TRUE(feature_test_main(
//       /*mlir_file_path*/ c_ft_path + "max_col_s_f32.mlir",
//       /*backend_types*/
//       kSupportedBackendList,
//       /*num_inputs*/ 1,
//       /*num_outputs*/ 1,
//       /*input_descriptors*/ {"110x100x13xf32_X"},
//       /*output_descriptors*/ {"f32_X"}));
//       // /*output_descriptors*/ {"f32_X"}, {input_val}));
//   unsetenv("DISC_ENABLE_STITCH");
//   unsetenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL");
//   unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
// }

// // dynamic shape 3D column reduction test case
// TEST(TFMaxOpTest, ColReduceFullyDynamicShape3DF32) {
//   setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
//   setenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", "true", 1);
//   setenv("DISC_ENABLE_STITCH", "true", 1);
//   EXPECT_TRUE(feature_test_main(
//       /*mlir_file_path*/ c_ft_path + "max_col_d_f32.mlir",
//       /*backend_types*/
//       kSupportedBackendList,
//       /*num_inputs*/ 1,
//       /*num_outputs*/ 1,
//       /*input_descriptors*/ {"110x100x13xf32_X"},
//       /*output_descriptors*/ {"f32_X"}));
//   unsetenv("DISC_ENABLE_STITCH");
//   unsetenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL");
//   unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
// }

// // dynamic shape 3D column reduction test case
// TEST(TFMaxOpTest, ColReduceFullyDynamicShape3DF32_1) {
//   setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
//   setenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", "true", 1);
//   setenv("DISC_ENABLE_STITCH", "true", 1);
//   EXPECT_TRUE(feature_test_main(
//       /*mlir_file_path*/ c_ft_path + "max_col_d_f32.mlir",
//       /*backend_types*/
//       kSupportedBackendList,
//       /*num_inputs*/ 1,
//       /*num_outputs*/ 1,
//       /*input_descriptors*/ {"512x100x10xf32_X"},
//       /*output_descriptors*/ {"f32_X"}));
//   unsetenv("DISC_ENABLE_STITCH");
//   unsetenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL");
//   unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
// }

// // dynamic shape 3D column reduction test case
// TEST(TFMaxOpTest, ColReduceFullyDynamicShape3DF32_2) {
//   setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
//   setenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", "true", 1);
//   setenv("DISC_ENABLE_STITCH", "true", 1);
//   EXPECT_TRUE(feature_test_main(
//       /*mlir_file_path*/ c_ft_path + "max_col_d_f32.mlir",
//       /*backend_types*/
//       kSupportedBackendList,
//       /*num_inputs*/ 1,
//       /*num_outputs*/ 1,
//       /*input_descriptors*/ {"8192x256x10xf32_X"},
//       /*output_descriptors*/ {"f32_X"}));
//   unsetenv("DISC_ENABLE_STITCH");
//   unsetenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL");
//   unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
// }

// // dynamic shape 3D column reduction test case
// TEST(TFMaxOpTest, ColReduceFullyDynamicShape3DF32_3) {
//   setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
//   setenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", "true", 1);
//   setenv("DISC_ENABLE_STITCH", "true", 1);
//   EXPECT_TRUE(feature_test_main(
//       /*mlir_file_path*/ c_ft_path + "max_col_d_f32.mlir",
//       /*backend_types*/
//       kSupportedBackendList,
//       /*num_inputs*/ 1,
//       /*num_outputs*/ 1,
//       /*input_descriptors*/ {"2560x8192x10xf32_X"},
//       /*output_descriptors*/ {"f32_X"}));
//   unsetenv("DISC_ENABLE_STITCH");
//   unsetenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL");
//   unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
// }

// // dynamic shape 3D column reduction test case
// TEST(TFMaxOpTest, ColReduceFullyDynamicShape3DF32_4) {
//   setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
//   setenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", "true", 1);
//   setenv("DISC_ENABLE_STITCH", "true", 1);
//   EXPECT_TRUE(feature_test_main(
//       /*mlir_file_path*/ c_ft_path + "max_col_d_f32.mlir",
//       /*backend_types*/
//       kSupportedBackendList,
//       /*num_inputs*/ 1,
//       /*num_outputs*/ 1,
//       /*input_descriptors*/ {"8192x8192x10xf32_X"},
//       /*output_descriptors*/ {"f32_X"}));
//   unsetenv("DISC_ENABLE_STITCH");
//   unsetenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL");
//   unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
// }

}  // namespace mlir_test
