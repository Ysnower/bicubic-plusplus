import torch
from training import dataset
from torch.utils.data import DataLoader
import yaml
from models.SR_models import Bicubic_plus_plus
from torchmetrics import Metric
from utils.img_utils import shave, rgb_to_ycbcr
from utils.ssim import SSIM
from torch.optim.lr_scheduler import StepLR


class MeanPSNR(Metric):
    def __init__(self):
        super(MeanPSNR, self).__init__()
        self.add_state("psnrs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, hr_image, sr_image):
        self.psnrs += torch.mean(10 * torch.log10(1 / (torch.mean(torch.square(hr_image - sr_image), (1, 2, 3)))))
        self.total += 1

    def compute(self):
        val = self.psnrs.float() / self.total
        return val


class MeanSSIM(Metric):
    def __init__(self, channel_num, data_range=1.0):
        super(MeanSSIM, self).__init__()
        self.add_state("ssims", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.ssim_calc = SSIM(data_range=data_range,
                              channel=channel_num)

    def update(self, hr_image, sr_image):
        self.ssims += self.ssim_calc(hr_image, sr_image)
        self.total += 1

    def compute(self):
        val = self.ssims.float() / self.total
        return val


class val(object):
    def __init__(self, conf):
        super(val, self).__init__()
        self.conf = conf
        self.use_Y = self.conf['trainer']['use_Y_channel_in_val']
        self.scale = self.conf['data']['train']['scale']
        self.psnr = MeanPSNR()
        self.ssim = MeanSSIM(channel_num=1 if self.conf['trainer']['use_Y_channel_in_val'] else 3, data_range=1.0)

    def update_val_metrics(self, sr_images, hr_images):
        shaved_sr = shave(sr_images, self.scale)
        shaved_hr = shave(hr_images, self.scale)
        if self.use_Y:
            shaved_sr = rgb_to_ycbcr(shaved_sr)
            shaved_hr = rgb_to_ycbcr(shaved_hr)
        self.psnr.update(torch.clamp(shaved_sr, 0, 1),
                         torch.clamp(shaved_hr, 0, 1))
        self.ssim.update(torch.clamp(shaved_sr, 0, 1),
                         torch.clamp(shaved_hr, 0, 1))


def cast_values(data):
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively call the function for nested dictionaries
            data[key] = cast_values(value)
        elif isinstance(value, str):
            if value == 'None':
                data[key] = None
            else:  # Convert to float if possible (ex. 10e-3)
                try:
                    data[key] = float(value)
                except ValueError:
                    pass
        elif isinstance(value, list):
            continue
    return data


def get_config(path='configs/conf.yaml'):
    with open(path, 'r') as file:
        conf = yaml.safe_load(file)
    conf = cast_values(conf)
    return conf


conf = get_config(path='configs/conf.yaml')


def train_dataloader(conf):
    ds = dataset.SRDataset1(conf_data=conf['data']['train'],
                            conf_deg=conf['degradation']['train'], is_train=conf['data']['train']["is_train"])
    loader = DataLoader(dataset=ds, **conf['loader']['train'])
    return loader


def val_dataloader(conf):
    ds = dataset.SRDataset1(conf_data=conf['data']['val'],
                            conf_deg=conf['degradation']['val'], is_train=conf['data']['val']["is_train"])
    loader = DataLoader(dataset=ds, **conf['loader']['val'])
    return loader


train_data = train_dataloader(conf)
val_data = val_dataloader(conf)
network = Bicubic_plus_plus(sr_rate=conf["network"]["params"]["sr_rate"]).cuda()
if conf["load_pretrained"]:
    network.load_state_dict(torch.load(conf["pretrained_path"]))
    print("Load pretrained model:{}".format(conf["pretrained_path"]))
optimizer = torch.optim.Adam(network.parameters(), lr=conf["trainer"]["base_lr_rate"])
scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.9)
iters = 0
val_func = val(conf=conf)
for i in range(1, conf["trainer"]["num_epochs"] + 1):
    network.train()
    for k, return_dict in enumerate(train_data):
        iters += 1
        img_lr = return_dict['img_lr']
        img_lr = img_lr.cuda()
        img_hr = return_dict['img_hr']
        img_hr = img_hr.cuda()
        network.zero_grad()
        img_sr = network(img_lr)
        train_loss = torch.mean(torch.square(img_sr - img_hr))
        train_loss = train_loss.sum()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if iters % 20 == 0:
            print("epoch:{}  iters:{}  loss:{:.5f}  lr:{:.5f}".format(i, iters, train_loss.item(),
                                                                      optimizer.param_groups[0]['lr']))
    scheduler.step()
    if i % conf["trainer"]["check_val_every_n_epoch"] == 0:
        print("......start eval......")
        network.eval()
        avg_psnr = 0
        avg_ssim = 0
        for j, val_dict in enumerate(val_data):
            val_lr = val_dict['img_lr']
            val_lr = val_lr.cuda()
            val_hr = val_dict['img_hr']
            val_hr = val_hr.cpu()
            val_sr = network(val_lr).cpu()
            val_func.update_val_metrics(val_sr, val_hr)
            val_psnr = val_func.psnr.compute()
            val_ssim = val_func.ssim.compute()
            avg_psnr += val_psnr
            avg_ssim += val_ssim
        print("avg_psnr:{}  avg_ssim:{}".format(avg_psnr / len(val_data), avg_ssim / len(val_data)))
        torch.save(network.state_dict(), conf["save_dir"] + "bicubic_PP_{}_{}.pth".format(i, avg_psnr / len(val_data)))
