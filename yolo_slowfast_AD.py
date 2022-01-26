from sys import maxsize
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from collections import defaultdict
import os,cv2,time,torch,pytorchvideo,warnings,argparse,math,json
warnings.filterwarnings("ignore",category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort
from sklearn.metrics import roc_auc_score


def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img

def ava_inference_transform(clip, boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def deepsort_update(Tracker,pred,xywh,np_img):
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    return outputs

def score_normalize(score, inverse=False):
    score=np.array(score,np.float16)
    _range=np.max(score)-np.min(score)
    score=(score-np.min(score))/(_range+1e-5) if not inverse else (np.max(score)-score)/(_range+1e-5)
    return score.tolist()

def score_moving_avg(score,K=0.35):
    for i in range(1,len(score)):
        score[i]=K*score[i]+(1-K)*score[i-1]
    return score

def count_labels_intrain(yolo_preds,id_to_ava_labels,label_counter):
    for i, (im, pred) in enumerate(zip(yolo_preds.imgs, yolo_preds.pred)):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if pred.shape[0]:
            for j, (*box, conf, cls, trackid, vx, vy) in enumerate(pred):
                label_counter["coco_label"][yolo_preds.names[int(cls)]]+=1
                if int(cls) == 0 and trackid in id_to_ava_labels:
                    for avaname in id_to_ava_labels[trackid]:
                        label_counter["ava_label"][avaname.split(' ')[0]]+=1
    return label_counter

def calculate_labels_score_intest(yolo_preds,id_to_ava_labels,label_prob,score_list):
    for i, (im, pred) in enumerate(zip(yolo_preds.imgs, yolo_preds.pred)):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        max_score=0
        if pred.shape[0]:
            for j, (*box, conf, cls, trackid, vx, vy) in enumerate(pred):
                prob_object=label_prob["coco_label"][yolo_preds.names[int(cls)]]
                prob_action=0
                if int(cls) == 0 and trackid in id_to_ava_labels:
                    for avaname in id_to_ava_labels[trackid]:
                        prob_action+=label_prob["ava_label"][avaname.split(' ')[0]]
                max_score=max(max_score,(1-prob_object)*(1-prob_action))
        if score_list:
            score_list.append(0.7*score_list[-1]+0.3*max_score)
        else:
            score_list.append(max_score)
    return score_list

def train(config):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 200
    if config.classes:
        model.classes = config.classes
    device = config.device
    imsize = config.imsize
    video_model = slowfast_r50_detection(True).eval().to(device)
    ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    label_counter={"coco_label":defaultdict(int),"ava_label":defaultdict(int)}
    
    a=time.time()
    for video_name in tqdm(natsorted(os.listdir(os.path.join(config.dataset,"training","videos"))),"training process"):
        video_path=os.path.join(config.dataset,"training","videos",video_name)
        deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
        video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
        for i in range(0,math.ceil(video.duration),1):
            video_clips=video.get_clip(i, i+1-0.04)
            video_clips=video_clips['video']
            if video_clips is None:
                continue
            img_num=video_clips.shape[1]
            imgs=[]
            for j in range(img_num):
                imgs.append(tensor_to_numpy(video_clips[:,j,:,:]))
            yolo_preds=model(imgs, size=imsize)
            yolo_preds.files=[f"img_{i*25+k}.jpg" for k in range(img_num)]
            deepsort_outputs=[]
            for j in range(len(yolo_preds.pred)):
                temp=deepsort_update(deepsort_tracker,yolo_preds.pred[j].cpu(),yolo_preds.xywh[j][:,0:4].cpu(),yolo_preds.imgs[j])
                if len(temp)==0:
                    temp=np.ones((0,8))
                deepsort_outputs.append(temp.astype(np.float32))
            yolo_preds.pred=deepsort_outputs
            id_to_ava_labels={}
            if yolo_preds.pred[img_num//2].shape[0]:
                inputs,inp_boxes,_=ava_inference_transform(video_clips,yolo_preds.pred[img_num//2][:,0:4],crop_size=imsize)
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()
                    _, top_classes = torch.topk(slowfaster_preds, k=3)
                    top_classes=top_classes.tolist()
                for tid,avalabels in zip(yolo_preds.pred[img_num//2][:,5].tolist(),top_classes):
                    id_to_ava_labels[tid]=[ava_labelnames[avalabel+1] for avalabel in avalabels]
            label_counter=count_labels_intrain(yolo_preds,id_to_ava_labels,label_counter)
        
    with open("exp/{}_{}_counter.json".format(os.path.basename(config.dataset),config.imsize),'w') as f:    
        data=json.dumps(label_counter,indent=2)
        f.write(data)
    print("train cost: {:.3f}s.".format(time.time()-a))

def test(config):
    with open("exp/{}_{}_counter.json".format(os.path.basename(config.dataset),config.imsize),'r') as f:    
        counter=json.load(f)
    label_prob={"coco_label":defaultdict(int),"ava_label":defaultdict(int)}
    for key in label_prob.keys():
        total_num=0
        for name in counter[key].keys():
            total_num+=counter[key][name]
        for name in counter[key].keys():
            label_prob[key][name]=counter[key][name]/total_num
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 200
    if config.classes:
        model.classes = config.classes
    device = config.device
    imsize = config.imsize
    video_model = slowfast_r50_detection(True).eval().to(device)
    ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    
    pred_list=[]
    a=time.time()
    for video_name in tqdm(natsorted(os.listdir(os.path.join(config.dataset,"testing","videos"))),"testing process"):
        video_path=os.path.join(config.dataset,"testing","videos",video_name)
        deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
        video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
        video_ans=[]
        for i in range(0,math.ceil(video.duration),1):
            video_clips=video.get_clip(i, i+1-0.04)
            video_clips=video_clips['video']
            if video_clips is None:
                continue
            img_num=video_clips.shape[1]
            imgs=[]
            for j in range(img_num):
                imgs.append(tensor_to_numpy(video_clips[:,j,:,:]))
            yolo_preds=model(imgs, size=imsize)
            yolo_preds.files=[f"img_{i*25+k}.jpg" for k in range(img_num)]
            deepsort_outputs=[]
            for j in range(len(yolo_preds.pred)):
                temp=deepsort_update(deepsort_tracker,yolo_preds.pred[j].cpu(),yolo_preds.xywh[j][:,0:4].cpu(),yolo_preds.imgs[j])
                if len(temp)==0:
                    temp=np.ones((0,8))
                deepsort_outputs.append(temp.astype(np.float32))
            yolo_preds.pred=deepsort_outputs
            id_to_ava_labels={}
            if yolo_preds.pred[img_num//2].shape[0]:
                inputs,inp_boxes,_=ava_inference_transform(video_clips,yolo_preds.pred[img_num//2][:,0:4],crop_size=imsize)
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()
                    _, top_classes = torch.topk(slowfaster_preds, k=3)
                    top_classes=top_classes.tolist()
                for tid,avalabels in zip(yolo_preds.pred[img_num//2][:,5].tolist(),top_classes):
                    id_to_ava_labels[tid]=[ava_labelnames[avalabel+1] for avalabel in avalabels]
            video_ans=calculate_labels_score_intest(yolo_preds,id_to_ava_labels,label_prob,video_ans)
        pred_list.extend(video_ans)
    

    select_path={"SHTech":"frame_labels_shanghai.npy",
                 "Ped2":"frame_labels_ped2.npy",
                 "Avenue":"frame_labels_avenue.npy"}
    gt_path=os.path.join("/data/VAD",select_path[os.path.basename(config.dataset)])
    gt=np.load(gt_path)
    if not gt_path.endswith('shanghai.npy'):
        gt=gt[0]
    
    if len(gt)!=len(pred_list):
        print("gt and pred's length do not match({},{}), we choose the short one.".format(len(gt),len(pred_list)))
    L=min(len(gt),len(pred_list))
    auc=roc_auc_score(gt[0:L],pred_list[0:L])
    
    print("auc:",round(auc,5))
    print("test cost: {:.3f}s.".format(time.time()-a))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/data/VAD/SHTech')
    # object detect config
    parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    config = parser.parse_args()
    if not os.path.exists("exp/{}_{}_counter.json".format(os.path.basename(config.dataset),config.imsize)):
        train(config)
    test(config)




