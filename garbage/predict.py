# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np
import cv2
import time

import sys
sys.path.append(r"D:\\project\\python\\dia_bot\\garbage")
sys.path.insert(0, ".")
import utils 
from utils import get_image_list


def predict(args, predictor):
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    test_num = 500
    test_time = 0.0
    if not args.enable_benchmark:
        # for PaddleHubServing
        if args.hubserving:
            img_list = [args.image_file]
        # for predict only
        else:
            img_list = get_image_list(args.image_file)

        for idx, img_name in enumerate(img_list):
            if not args.hubserving:
                img = cv2.imread(img_name)[:, :, ::-1]
                assert img is not None, "Error in loading image: {}".format(
                    img_name)
            else:
                img = img_name
            inputs = utils.preprocess(img, args)
            inputs = np.expand_dims(
                inputs, axis=0).repeat(
                    args.batch_size, axis=0).copy()
            input_tensor.copy_from_cpu(inputs)

            predictor.run()

            output = output_tensor.copy_to_cpu()
            classes, scores = utils.postprocess(output, args)
            if args.hubserving:
                return classes, scores
            print("Current image file: {}".format(img_name))
            print("\ttop-1 class: {0}".format(classes[0]))
            print("\ttop-1 score: {0}".format(scores[0]))
            result = [classes,scores]
            # return classes, scores
            return result

    else:
        for i in range(0, test_num + 10):
            inputs = np.random.rand(args.batch_size, 3, 224,
                                    224).astype(np.float32)
            start_time = time.time()
            input_tensor.copy_from_cpu(inputs)

            predictor.run()

            output = output_tensor.copy_to_cpu()
            output = output.flatten()
            if i >= 10:
                test_time += time.time() - start_time
            time.sleep(0.01)  # sleep for T4 GPU

        fp_message = "FP16" if args.use_fp16 else "FP32"
        trt_msg = "using tensorrt" if args.use_tensorrt else "not using tensorrt"
        print("{0}\t{1}\t{2}\tbatch size: {3}\ttime(ms): {4}".format(
            args.model, trt_msg, fp_message, args.batch_size, 1000 * test_time
            / test_num))


def main(image_path):
    dic_a = {
        "batch_size": 1, 
        "class_num":1000, 
        "cpu_num_threads":10, 
        "enable_benchmark":False, 
        "enable_mkldnn":False, 
        "enable_profile":False, 
        "gpu_mem":8000, 
        "hubserving":False, 
        "image_file":'image_path', 
        "ir_optim":True, 
        "load_static_weights":False, 
        "model":None, 
        "model_file":'D:\\project\\python\\dia_bot\\garbage\\inference\\inference.pdmodel', 
        "normalize":True, 
        "params_file":'D:\\project\\python\\dia_bot\\garbage\\inference\\inference.pdiparams', 
        "pre_label_image":False, 
        "pre_label_out_idr":None, 
        "pretrained_model":None, 
        "resize":224, 
        "resize_short":256, 
        "top_k":1, 
        "use_fp16":False, 
        "use_gpu" : False, 
        "use_tensorrt": False,
    }

    dic_a['image_file'] = image_path

    # 创建 ArgumentParser() 对象
    parser = argparse.ArgumentParser()
    # 调用 add_argument() 方法添加参数
    # parser.add_argument("-a")
    # 使用 parse_args() 解析添加的参数
    args = argparse.Namespace(**dic_a)

    print()
    print(args)
    print(type(args))
    if not args.enable_benchmark:
        assert args.batch_size == 1
    else:
        assert args.model is not None
    # HALF precission predict only work when using tensorrt
    if args.use_fp16 is True:
        assert args.use_tensorrt is True

    predictor = utils.create_paddle_predictor(args)
    result = predict(args, predictor)
    return result
    

# if __name__ == "__main__":
    
#     main('D:\\project\\python\\dia_bot\\The-Eye-Konws-the-Garbage\\picture\\1866632657810606907.jpg')
