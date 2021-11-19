import numpy as np
import cv2
from PIL import Image 
import pytesseract
import os

dizin="Dataset"
imgs=os.listdir(dizin)
uzunluk=len(imgs)

mail=[]
control=[]
for i in range(len(imgs)):
    mail.append(imgs[i].split("_")[0])
    control.append(imgs[i].split("_")[1].strip(".png"))
    

newImages=[]
for j in imgs:
    images=cv2.imread(f'{dizin}/{j}')
    images.shape
    # images=cv2.resize(images,(1280,1920))
    # images=cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    # images_filter=np.array([[-1,-1,-1],
    #                         [-1,9,-1],
    #                         [-1,-1,-1]])
    # sharpened_images=cv2.filter2D(images,-1,images_filter)
    newImages.append(images)
    # cv2.imshow("images",images)
    # cv2.waitKey(0)
    


newMailList=[]
for i in newImages:
    text=pytesseract.image_to_string(i,lang="tur")  
    # newMailList.append(text)
    newMailList.append(text.replace("\n" , ","))
    
    
# # csv olarak kaydet.
# import csv
# with open("dataset.csv","w",newline="",encoding="UTF-8") as file:
#     writer=csv.writer(file)    
#     writer.writerow(["mail" , "control"])
    
#     for idx in newMailList:
        
#         writer.writerow(idx.split(","))
        
        
#%% 
import pandas as pd

data = {'Mails': newMailList,
        'control': control}

	
#dataframe oluştur
df = pd.DataFrame(data)
a=df.to_csv("dataset.csv",encoding="UTF-8")



# compression_opts = dict(method='zip',archive_name='out.csv')
# df.to_csv('out.zip', index=False,
#           compression=compression_opts)


#%% tek resimlik denemeler
import cv2
import pytesseract
img=cv2.imread("9360.png")
text=pytesseract.image_to_string(img,lang="tur")


