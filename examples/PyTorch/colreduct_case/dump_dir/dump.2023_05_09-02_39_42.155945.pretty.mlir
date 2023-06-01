#loc = loc(unknown)
module {
  func.func @main(%arg0: tensor<8192x2560xf32> loc(unknown), %arg1: tensor<8192x2560xf32> loc(unknown)) -> tensor<?xf32> attributes {tf.entry_function = {input_placements = "gpu,gpu", inputs = "arg0_1.1_,arg1_1.1_", output_placements = "gpu", outputs = "sum_1.1"}} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc1)
    %1 = chlo.broadcast_subtract %arg0, %arg1 : (tensor<8192x2560xf32>, tensor<8192x2560xf32>) -> tensor<?x?xf32> loc(#loc2)
    %2 = chlo.broadcast_multiply %1, %arg0 : (tensor<?x?xf32>, tensor<8192x2560xf32>) -> tensor<?x?xf32> loc(#loc3)
    %3 = mhlo.reduce(%2 init: %0) applies mhlo.add across dimensions = [0] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc(#loc1)
    return %3 : tensor<?xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("<eval_with_key>.5":7:12)
#loc2 = loc("<eval_with_key>.5":5:10)
#loc3 = loc("<eval_with_key>.5":6:10)
