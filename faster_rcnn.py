import torchvision,torch,cv2,random,time
import numpy as np

coco_names=[]
with open('coco_names.txt',encoding='utf-8') as f:
    for line in f:
        coco_names.append(line.replace("\n",""))

def img_preprocessing(np_img,device):
    np_img = np_img[:, :, ::-1].transpose(2, 0, 1)
    np_img = np.ascontiguousarray(np_img)
    tensor_img=torch.from_numpy(np_img).to("cuda:{}".format(device))
    tensor_img=tensor_img[np.newaxis,:].float()
    tensor_img /= 255.0
    return tensor_img

def plot_one_box(x, img, color=[100,100,100], text_info="None",thickness=2,fontsize=0.5,fontthickness=1):
    # Plots one bounding box on image img
    
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    
    t_size=cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.4)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    
    return img

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
for name,parameter in model.named_parameters():
    print(name,parameter.shape)
model.to('cuda').eval()

img=cv2.imread("test_img/input/input.jpg")
print(img.shape)

a=time.time()
x = img_preprocessing(img,'0')
print("preprocess cost: {:.5f}".format(time.time()-a))


b=time.time()
with torch.no_grad():
    pred,features=model(x)
print("inference cost: {:.5f}".format(time.time()-b))
    
c=time.time()
pred=pred[0]
print(pred['boxes'])
print(pred['labels'])
for i in range(pred['labels'].shape[0]):
    color=[random.randint(0, 255) for _ in range(3)]
    text='{}|{:.2f}'.format(coco_names[pred['labels'][i]-1],pred['scores'][i].item())
    img=plot_one_box(pred['boxes'][i,:],img,color,text_info=text)
cv2.imwrite("test_img/output/output.jpg",img)
print("post process cost: {:.5f}".format(time.time()-c))

print("total cost: {:.5f}".format(time.time()-a))


    