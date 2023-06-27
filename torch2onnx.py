import torch
from models.SR_models import Bicubic_plus_plus


torch.set_grad_enabled(False)
torch_model = Bicubic_plus_plus(sr_rate=3)
model_path = "pretrained/bicubic_pp_x3.pth"
torch_model.load_state_dict(torch.load(model_path), False)
batch_size = 1
input_shape = (3, 128, 128)
# set the model to inference mode
torch_model.eval().cuda()
x = torch.randn(batch_size, *input_shape).cuda()
export_onnx_file = "out.onnx"
torch.onnx.export(torch_model,
                  x,
                  export_onnx_file,
                  opset_version=13,
                  do_constant_folding=True,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"},
                                "output": {0: "batch_size"}})
