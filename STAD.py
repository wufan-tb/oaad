import warnings
warnings.filterwarnings("ignore",category=UserWarning)

import os,torch,cv2,pytorchvideo,time,argparse,json
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from sklearn.metrics import roc_auc_score

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection # Another option is slowfast_r50_detection, slow_r50_detection

from self_utils.visualization import VideoVisualizer

# ## Define the transformations for the input required by the model
def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):

    boxes = np.array(boxes)
    roi_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )
    
    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )
    
    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )
    
    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def score_normalize(score, inverse=False):
    score=np.array(score,np.float16)
    _range=np.max(score)-np.min(score)
    score=(score-np.min(score))/(_range+1e-5) if not inverse else (np.max(score)-score)/(_range+1e-5)
    return score.tolist()

def score_moving_avg(score,K=0.35):
    for i in range(1,len(score)):
        score[i]=K*score[i]+(1-K)*score[i-1]
    return score

def get_model(device='cuda',threshold=0.8):
    # ## load slow faster model
    video_model = slowfast_r50_detection(True) # Another option is slowfast_r50_detection
    video_model = video_model.eval().to(device)

    # ## Load an off-the-shelf Detectron2 object detector
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    print("completed loading pre-trained models.")
    return predictor,video_model

def load_label_names(coco_label_file='self_utils/coco_names.txt',ava_label_file='self_utils/ava_action_list.pbtxt'):
    coco_labels={}
    with open(coco_label_file,encoding='utf-8') as f:
        index=0
        for line in f:
            coco_labels[index]=line.replace("\n","")
            index+=1
    ava_labels,_=AvaLabeledVideoFramePaths.read_label_map(ava_label_file)
    return coco_labels,ava_labels

def save_video(gif_imgs,vide_save_path='test_img/video/output.mp4'):
    # ## Save predictions as video
    height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]
    video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))

    for image in gif_imgs:
        img = (255*image).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    video.release()
    print('Predictions are saved to the video file: ', vide_save_path)
    
# This method takes in an image and generates the bounding boxes for people in the image.
def get_object_bboxes(inp_img, predictor):
    with torch.no_grad():
        predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    # print("ROI info:",boxes.tensor.shape,scores.shape,predictions.pred_classes.shape)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.7 )].tensor.cpu() # select only person
    return predicted_boxes,predictions.pred_classes.detach().tolist(),scores.detach().tolist()

def train(args):
    predictor,video_model=get_model(args.device,args.threshold)
    # Create an id to label name mapping
    coco_labels,ava_labels = load_label_names()
    label_counter={"coco_label":{},"ava_label":{}}
    a=time.time()
    for video_name in tqdm(natsorted(os.listdir(os.path.join(args.dataset,"training","videos"))),"training process"):
        video_path=os.path.join(args.dataset,"training","videos",video_name)
        encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
        time_stamp_range = range(0,int(encoded_vid.duration//1),1) # time stamps in video for which clip is sampled. 
        clip_duration = 1 # Duration of clip used for each inference step.
        for time_stamp in time_stamp_range:
            # Generate clip around the designated time stamps
            inp_imgs = encoded_vid.get_clip(time_stamp ,time_stamp + clip_duration - 0.04 )
            inp_imgs = inp_imgs['video']
            for i in range(0,inp_imgs.shape[1],args.sample_step):
                inp_img = inp_imgs[:,i,:,:]
                inp_img = inp_img.permute(1,2,0)
                _,object_labels,object_scores= get_object_bboxes(inp_img, predictor) 
                if i==((inp_imgs.shape[1]//args.sample_step)//2)*args.sample_step:
                    predicted_boxes=_
                for label,score in zip(object_labels,object_scores):
                    name=coco_labels[label]
                    if name not in label_counter["coco_label"]:
                        label_counter["coco_label"][name]=score
                    else:
                        label_counter["coco_label"][name]+=score

            if len(predicted_boxes) == 0: 
                continue      
            inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy(),crop_size=args.imsize)
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(args.device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(args.device)
            with torch.no_grad():
                preds = video_model(inputs, inp_boxes.to(args.device))
            preds = preds.to('cpu')
            # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
            preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)
            
            motion_scores,motion_labels = preds.max(-1)
            for score,label in zip(motion_scores.tolist(),motion_labels.tolist()):
                name=ava_labels[label]
                if score<0.5:
                    continue
                if name not in label_counter["ava_label"]:
                    label_counter["ava_label"][name]=score
                else:
                    label_counter["ava_label"][name]+=score
            
    with open("exp/{}_{}_counter.json".format(os.path.basename(args.dataset),args.imsize),'w') as f:    
        data=json.dumps(label_counter,indent=2)
        f.write(data)
    print("train cost: {:.3f}s.".format(time.time()-a))

def test(args):
    predictor,video_model=get_model(args.device,args.threshold)
    # Create an id to label name mapping
    coco_labels,ava_labels = load_label_names()
    with open("exp/{}_{}_counter.json".format(os.path.basename(args.dataset),args.imsize),'r') as f:    
        counter=json.load(f)
    label_prob={"coco_label":{},"ava_label":{}}
    for key in label_prob.keys():
        total_num=0
        for name in counter[key].keys():
            total_num+=counter[key][name]
        for name in counter[key].keys():
            label_prob[key][name]=counter[key][name]/total_num
    
    a=time.time()
    pred_list=[]
    for video_name in tqdm(natsorted(os.listdir(os.path.join(args.dataset,"testing","videos"))),"testing process"):
        video_path=os.path.join(args.dataset,"testing","videos",video_name)
        encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
        time_stamp_range = range(0,int(encoded_vid.duration//1),1) # time stamps in video for which clip is sampled. 
        clip_duration = 1 # Duration of clip used for each inference step.
        scores=[]
        for time_stamp in time_stamp_range:
            # Generate clip around the designated time stamps
            inp_imgs = encoded_vid.get_clip(time_stamp ,time_stamp + clip_duration - 0.04 )
            inp_imgs = inp_imgs['video']
            for i in range(0,inp_imgs.shape[1],args.sample_step):
                inp_img = inp_imgs[:,i,:,:]
                inp_img = inp_img.permute(1,2,0)
                _,label_list,score_list= get_object_bboxes(inp_img, predictor) 
                if i==((inp_imgs.shape[1]//args.sample_step)//2)*args.sample_step:
                    predicted_boxes=_
                temp=[-1]
                for label,score in zip(label_list,score_list):
                    name=coco_labels[label]
                    if name not in label_prob["coco_label"].keys():
                        temp.append(1)
                    else:
                        temp.append(1-label_prob["coco_label"][name])
                for _ in range(args.sample_step):
                    scores.append(max(temp))
            
            temp=[-1]
            if len(predicted_boxes) > 0: 
                inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy(),crop_size=args.imsize)
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(args.device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(args.device)
                with torch.no_grad():
                    preds = video_model(inputs, inp_boxes.to(args.device))
                preds = preds.to('cpu')
                # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
                preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)
                
                motion_scores,motion_labels = preds.max(-1)
                
                for score,label in zip(motion_scores.tolist(),motion_labels.tolist()):
                    if score<0.5:
                        continue
                    name=ava_labels[label]
                    if name not in label_prob["ava_label"].keys():
                        temp.append(0.5)
                    else:
                        temp.append(0.5-label_prob["ava_label"][name])
                        
            if time_stamp==int(encoded_vid.duration//1)-1:
                scores[time_stamp*25:]=[max(x,max(temp)) for x in scores[time_stamp*25:]]
                diff=round(25*encoded_vid.duration)-len(scores)
                if diff >= 0:
                    last=scores[-1]
                    scores.extend([last for _ in range(diff)])
                else:
                    for _ in range(0-diff):
                        scores.pop()
            else:
                1+1
                scores[time_stamp*25:time_stamp*25+25]=[max(x,max(temp)) for x in scores[time_stamp*25:time_stamp*25+25]]
               
        pred_list+=score_normalize(score_moving_avg(scores))

    select_path={"SHTech":"frame_labels_shanghai.npy",
                 "Ped2":"frame_labels_ped2.npy",
                 "Avenue":"frame_labels_avenue.npy"}
    gt_path=os.path.join("/data/VAD",select_path[os.path.basename(args.dataset)])
    gt=np.load(gt_path)
    if not gt_path.endswith('shanghai.npy'):
        gt=gt[0]
    
    assert len(gt)==len(pred_list),"{},{}".format(len(gt),len(pred_list))
    auc=roc_auc_score(gt,pred_list)
    
    print("auc:",round(auc,5))
    print("test cost: {:.3f}s.".format(time.time()-a))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/data/VAD/SHTech')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--imsize', type=int, default=640)
    parser.add_argument('--sample_step', type=int, default=5)
    
    args = parser.parse_args()
    if not os.path.exists("exp/{}_{}_counter.json".format(os.path.basename(args.dataset),args.imsize)):
        train(args)
    test(args)




