import torch
import yaml
import onnx
import onnxsim
from model.TSN.YOWOv3 import build_yowov3
import warnings
warnings.filterwarnings("ignore")

name = "falldown_best"
output_onnx_path = f"{name}.onnx"
model_weight = f"{name}.pth"

def build_config():
    ucf_config_file = f'utils/YAML/{name}.yaml'
    print(f"config_file: {ucf_config_file}")
    with open(ucf_config_file, "r") as file:
        return yaml.load(file, Loader=yaml.SafeLoader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

config = build_config()
config["pretrain_path"] = model_weight
model = build_yowov3(config)
print(f"model_weight: {config['pretrain_path']}")

imgh_size = config['img_size']
clip_length = config['clip_length']
model.to(device).eval()

dummy_input = torch.zeros(1, 3, clip_length, imgh_size[0], imgh_size[1], device=device)
print(f"dummy_input: {dummy_input.shape}")

# Export
print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    output_onnx_path,
    verbose=False,
    input_names=['Inputs'],
    output_names=['Outputs'],
    dynamo=False,  # don't save .data file 
    export_params=True,
    opset_version=19,
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
print(f"✓ Exported: {output_onnx_path}")

# Load & verify (merges any .data sidecar)
print("\nSimplifying ONNX model...")
onnx_model = onnx.load(output_onnx_path, load_external_data=True)
onnx.checker.check_model(onnx_model)

try:
    print("Starting simplification...")
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "simplify check failed"
except Exception as e:
    print(f"Simplifier failure: {e}")

# Save as single file (no .data sidecar)
onnx.save(onnx_model, output_onnx_path, save_as_external_data=False)
print(f"✓ Saved: {output_onnx_path}")