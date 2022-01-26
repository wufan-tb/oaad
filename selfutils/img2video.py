import os,cv2
from natsort import natsorted


def main(dataset_name):
    for phase in ["training", "testing"]:
        dataset_path="/data/VAD/{}/{}/frames".format(dataset_name,phase)
        for folder_name in natsorted(os.listdir(dataset_path)):
            first=os.listdir(os.path.join(dataset_path,folder_name))[0]
            img=cv2.imread(os.path.join(dataset_path,folder_name,first))
            width,height=img.shape[1],img.shape[0]
            video_path="/data/VAD/{}/{}/videos".format(dataset_name,phase)
            os.makedirs(video_path,exist_ok=True)
            save_path=os.path.join(video_path,"{}.mp4".format(folder_name))
            video = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))
            for img_name in natsorted(os.listdir(os.path.join(dataset_path,folder_name))):
                img = cv2.imread(os.path.join(dataset_path,folder_name,img_name))
                video.write(img)
            video.release()
            
if __name__=="__main__":
    for dataset_name in ["Ped2","Avenue","SHTech"]:
        main(dataset_name)
        print("{} data done.".format(dataset_name))