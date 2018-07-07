import torch
import numpy as np
import argparse

from models import FlowNet2,FlowNet2C #the path is depended on where you create this module
from utils.frame_utils import read_gen#the path is depended on where you create this module 
from datasets import StaticRandomCrop
from torch.autograd import Variable

if __name__ == '__main__':
    model_path="/home/data/flownet2_data/pre-trained_model/FlowNet2_checkpoint.pth"
    pim1_path="/home/data/flownet2_data/test/clean/ambush_1/frame_0001.png"
    pim2_path="/home/data/flownet2_data/test/clean/ambush_1/frame_0002.png"
    flow_path="tmp.flo"
    
    #obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
    args = parser.parse_args()
    args.grads = {}
    #initial a Net
    net = FlowNet2(args).cuda()
    #load the state_dict
    dict = torch.load(model_path)
    net.load_state_dict(dict["state_dict"])
    
    #load the image pair, you can find this operation in dataset.py
    pim1 = read_gen(pim1_path)
    pim2 = read_gen(pim2_path)
    images = [pim1, pim2]
    
    image_size = pim1.shape[:2]
    crop_size = args.crop_size
    cropper = StaticRandomCrop(image_size, crop_size)
    images = map(cropper, images)
    
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
    
    #process the image pair to obtian the flow 
    im=Variable(im)
    result = net(im).squeeze()

    #save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project 
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()

    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow(flow_path,data)