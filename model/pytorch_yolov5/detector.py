import torch,random
import numpy as np
from torch import tensor
try:
    from utils.datasets import letterbox
    from utils.utils import non_max_suppression
    from utils.utils import scale_coords,plot_one_box
except:
    from model.pytorch_yolov5.utils.datasets import letterbox
    from model.pytorch_yolov5.utils.utils import non_max_suppression
    from model.pytorch_yolov5.utils.utils import scale_coords,plot_one_box

def load_yolo_model(weights,device):
    model=torch.load(weights,map_location=lambda storage, loc: storage.cuda(int(device)))['model'].float().fuse().eval()
    names=model.module.names if hasattr(model, 'module') else model.names
    return model,names

def img_preprocessing(np_img,device,newsize=640):
    np_img=letterbox(np_img,new_shape=newsize)[0]
    np_img = np_img[:, :, ::-1].transpose(2, 0, 1)
    np_img = np.ascontiguousarray(np_img)
    tensor_img=torch.from_numpy(np_img).to("cuda:{}".format(device))
    tensor_img=tensor_img[np.newaxis,:].float()
    tensor_img /= 255.0
    return tensor_img

def yolov5_prediction(model,tensor_img,conf_thres,iou_thres,classes):
    with torch.no_grad():
        out,features=model(tensor_img)
        out=out[0]
        pred = non_max_suppression(out, conf_thres, iou_thres, classes=classes)[0]
    return pred,features

def post_processing(np_img,pred,class_names,inference_shape,class_colors=None):
    colors = class_colors if class_colors != None else [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(inference_shape[2:], pred[:, :4], np_img.shape).round()
        for *xyxy, conf, cls in pred:
            text_info = '%s,%.2f' % (class_names[int(cls)],conf)
            plot_one_box(xyxy, np_img, text_info=text_info, color=colors[int(cls)]) 

    return np_img,pred
