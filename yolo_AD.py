import sys,argparse,cv2,os,json,time
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from model.pytorch_yolov5.detector import load_yolo_model,img_preprocessing,yolov5_prediction,post_processing
sys.path.append('model/pytorch_yolov5/')

def normalize(score, inverse=False):
    score=np.array(score,np.float16)
    _range=np.max(score)-np.min(score)
    score=(score-np.min(score))/(_range+1e-5) if not inverse else (np.max(score)-score)/(_range+1e-5)
    return score.tolist()

def moving_avg(score,K=0.35):
    for i in range(1,len(score)):
        score[i]=K*score[i]+(1-K)*score[i-1]
    return score

def train(config):
    a=time.time()
    model,class_names=load_yolo_model(config.yolo_weights,config.device)
    counter={}
    for folder_name in tqdm(sorted(os.listdir(os.path.join(config.dataset,"training","frames"))),"training process"):
        for img_name in sorted(os.listdir(os.path.join(config.dataset,"training","frames",folder_name))):
            img=cv2.imread(os.path.join(config.dataset,"training","frames",folder_name,img_name))
            tensor_img=img_preprocessing(img,config.device)
            pred,features=yolov5_prediction(model,tensor_img,config.threshold,0.5,None)
            if pred is not None and len(pred):
                label_list=pred[:,-1].detach().flatten().tolist()
                score_list=pred[:,-2].detach().flatten().tolist()
                for label,score in zip(label_list,score_list):
                    name=class_names[int(label)]
                    if name not in counter:
                        counter[name]=score
                    else:
                        counter[name]+=score
                        
    with open("exp/{}_{}_counter.json".format(os.path.basename(config.dataset),config.threshold),'w') as f:    
        data=json.dumps(counter,indent=2)
        f.write(data)
    print("train cost: {:.3f}s.".format(time.time()-a))
    
def test(config):
    a=time.time()
    model,class_names=load_yolo_model(config.yolo_weights,config.device)
    with open("exp/{}_{}_counter.json".format(os.path.basename(config.dataset),config.threshold),'r') as f:    
        counter=json.load(f)
    total_num=0
    for name in counter.keys():
        total_num+=counter[name]
    print("total object num:",int(total_num))
    
    score_prob={}
    for name in counter.keys():
        score_prob[name]=counter[name]/total_num
        
    score_list=[]
    for folder_name in tqdm(sorted(os.listdir(os.path.join(config.dataset,"testing","frames"))),"testing process"):
        scores=[]
        for img_name in sorted(os.listdir(os.path.join(config.dataset,"testing","frames",folder_name))):
            img=cv2.imread(os.path.join(config.dataset,"testing","frames",folder_name,img_name))
            tensor_img=img_preprocessing(img,config.device)
            pred,features=yolov5_prediction(model,tensor_img,0.5,0.5,None)
            if pred is not None and len(pred):
                label_list=pred[:,-1].detach().flatten().tolist()
                temp=[]
                for label in label_list:
                    name=class_names[int(label)]
                    if name not in score_prob.keys():
                        temp.append(1)
                    else:
                        temp.append(1-score_prob[name]) 
                scores.append(max(temp))
            else:
                scores.append(-1)
        score_list+=normalize(moving_avg(scores))
        
    select_path={"SHTech":"frame_labels_shanghai.npy",
                 "Ped2":"frame_labels_ped2.npy",
                 "Avenue":"frame_labels_avenue.npy"
                 }
    gt_path=os.path.join("/data/VAD",select_path[os.path.basename(config.dataset)])
    
    gt=np.load(gt_path)
    if not gt_path.endswith('shanghai.npy'):
        gt=gt[0]
        
    auc=roc_auc_score(gt[0:len(score_list)],score_list)
    
    print("auc:",round(auc,5))
    print("test cost: {:.3f}s.".format(time.time()-a))
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/data/VAD/SHTech')
    parser.add_argument('--yolo_weights', type=str, default='model/pytorch_yolov5/weights/yolov5l.pt')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    
    config = parser.parse_args()
    print(config)
    
    json_path="exp/{}_counter.json".format(os.path.basename(config.dataset))
    if os.path.exists(json_path):
        test(config)
    else:
        train(config)
        test(config)
        
    