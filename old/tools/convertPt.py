from ultralytics import YOLO

m = YOLO("yolov11s_fp32.pt")
m.export(format="onnx", simplify=True, device="cpu")


exit()

from rknn.api import RKNN

# # Create RKNN object
rknn = RKNN()

# Load ONNX model
# INT8
print("--> Configuring model")
rknn.config(target_platform="rk3588", std_values=[255, 255, 255])
print("done")

dataset = "./dataset.txt"
modelPrefix = "yolov11s_fp32"

# rknn.hybrid_quantization_step2(model_input=f"{modelPrefix}.model",data_input=f"{modelPrefix}.data",model_quantization_cfg=f"{modelPrefix}.quantization.cfg")

print("--> Loading ONNX model")
ret = rknn.load_onnx(model=f"{modelPrefix}.onnx")
if ret != 0:
    print("Load ONNX model failed!")
    exit(ret)
print("done")

print("--> Building model")

# rknn.hybrid_quantization_step1(dataset=dataset)
# rknn.release()


ret = rknn.build(do_quantization=True, dataset=dataset)
# ret = rknn.build(do_quantization=False)
if ret != 0:
    print("Build model failed!")
    exit(ret)
print("done")

print("--> Exporting RKNN model")
rknn.export_rknn(f"./yolov11s_w8a16.rknn")
print("done")

rknn.release()
