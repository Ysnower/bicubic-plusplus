import torch
from models.SR_models import Bicubic_plus_plus
import cv2
from torchvision import transforms
import numpy as np


torch.set_grad_enabled(False)
model_path = "checkpoint/bicubic_PP_100_29.579639434814453.pth"
cuda = torch.cuda.is_available()
network = Bicubic_plus_plus(sr_rate=3)
if cuda:
    network = network.cuda()
network.load_state_dict(torch.load(model_path))
network.eval()
transform_data = transforms.Compose([transforms.ToTensor()])
img_path = "test_img/0004x3.png"
img = cv2.imread(img_path)
img = transform_data(img).unsqueeze(0)
img = img.cuda()
output = network(img)
output = output.squeeze().cpu().numpy()
output = output*255
output = np.transpose(output, (1, 2, 0))
cv2.imwrite("out_images/"+img_path.split('/')[-1], output)


