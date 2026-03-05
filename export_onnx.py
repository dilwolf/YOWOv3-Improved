import torch
import yaml
import onnx
import onnxsim
from model.TSN.YOWOv3 import build_yowov3
import warnings
warnings.filterwarnings("ignore")

name = "fire_best"
output_onnx_path = f"{name}_onnx.onnx"
model_weight = f"{name}.pth" 

def build_config():
    ucf_config_file = f'utils/YAML/{name}.yaml'
    print(f"config_file: {ucf_config_file}")
    with open(ucf_config_file, "r") as file:
        ucf_config = yaml.load(file, Loader=yaml.SafeLoader)    
    return ucf_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

config = build_config()
config["pretrain_path"] = str(model_weight)
model = build_yowov3(config)
pretrain_path = config["pretrain_path"]
print(f"model_weight: {pretrain_path}")
imgh_size = config['img_size']
clip_length = config['clip_length']
model.to(device)
model.eval()

dummy_input = torch.zeros(1, 3, clip_length, imgh_size[0], imgh_size[1])
print(f"dummy_input: {dummy_input.shape}")
dummy_input = dummy_input.to(device)

# Export to ONNX
print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    output_onnx_path,
    verbose=False,
    input_names=['Inputs'],
    output_names=['Outputs'],
    export_params=True,
    opset_version=17,
    do_constant_folding=False,
    keep_initializers_as_inputs=False,
    dynamic_axes={
        'Inputs': {
            0: 'batch_size',      # batch dimension
            # 3: 'height',          # image height
            # 4: 'width'            # image width
        },
        'Outputs': {
            0: 'batch_size'       # output also depends on batch size
        }
    }
)
print(f"✓ ONNX model exported as: {output_onnx_path}")

# Simplify ONNX model
print("\nSimplifying ONNX model...")
# Checks
onnx_model = onnx.load(output_onnx_path)  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model

try:
    print('\nStarting to simplify ONNX...')
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, 'assert check failed'
except Exception as e:
    print(f'Simplifier failure: {e}')

# print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
onnx.save(onnx_model, output_onnx_path)
print(f"ONNX export success, saved as {output_onnx_path}")