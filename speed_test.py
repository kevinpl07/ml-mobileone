import torch
import subprocess
import time


from openvino.runtime import Core
import segmentation_models_pytorch as smp
from mobileone import reparameterize_model
import numpy as np
import csv
ie = Core()
import onnxruntime as rt
entries = []
openvino = False
for backbone in ["mobileone_s0", "mobileone_s1", "mobileone_s2", "mobileone_s3", "mobileone_s4"]:
    for arch in ['unet', 'unetplusplus', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3plus', 'pan']:
    # for arch in ['unet', 'pan']:


        try:
            model = smp.create_model(arch, encoder_name=backbone, in_channels=3, classes=1)
            model.eval()

            print(backbone)
            print(arch)
            

            if "mobileone" in backbone:
                model = reparameterize_model(model)

            start = time.time()

            

            torch.onnx.export(model, torch.rand((1, 3, 256, 256)), f"models/{backbone}_{arch}_{256}.onnx", opset_version=11)


            if openvino:
                subprocess.call(["mo", "--input_model", f"models/{backbone}_{arch}_{256}.onnx", "--input_shape", f"[1, 3, {256}, {256}]", "--data_type", "FP16", "--output_dir", f"models/{backbone}_{arch}_{256}"])       
                model_ir = ie.read_model(model=f"models/{backbone}_{arch}_{256}/{backbone}_{arch}_{256}.xml")
                compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

                # Get output layer
                output_layer_ir = compiled_model_ir.output(0)

                # Run inference on the input image

                
                input_img = np.random.rand(1, 3, 256, 256).astype(np.float32)
            
                start = time.time()
                result = compiled_model_ir(input_img)[output_layer_ir]
                end = time.time()
                print(f"{arch}_{256}: {end-start}")
                entries.append([f"{backbone}_{arch}_{256}", end-start])
            else:
                input_img = np.random.rand(1, 3, 256, 256).astype(np.float32)
                
                sess = rt.InferenceSession(f"models/{backbone}_{arch}_{256}.onnx")
                input_name = sess.get_inputs()[0].name
                label_name = sess.get_outputs()[0].name
                start = time.time()
                pred = sess.run([label_name], {input_name: input_img})[0]
                end = time.time()
                print(f"{arch}_{256}: {end-start}")
                entries.append([f"{backbone}_{arch}_{256}", end-start])
        except Exception as e:
            entries.append([f"{backbone}_{arch}_{256}", -1])



with open('results_m1.csv', 'w', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(entries)