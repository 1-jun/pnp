#%%
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

#%%
def show_jpgs(
    subject_id = None,
    study_id = None,
    mimic_images_path = '/media/wonjun/New Volume1/MIMIC-CXR/MIMIC_CXR/files'
):
        
    images_path = os.path.join(mimic_images_path,
                            f'p{subject_id}'[:3],
                            f'p{subject_id}',
                            f's{study_id}')
    images = os.listdir(images_path)
    
    if len(images) > 1:
        fig, ax = plt.subplots(1, len(images))
        for i, _img_path in enumerate(images):
            img_path = os.path.join(images_path, _img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i].imshow(img)
        plt.show()
    elif len(images) == 1:
        fig, ax = plt.subplots()
        img_path = os.path.join(images_path, images[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        plt.show()
    else:
        print("zero images in that path")
        
def show_report(
    subject_id = None,
    study_id = None,
    mimic_reports_path = '/media/wonjun/New Volume1/MIMIC-CXR/MIMIC_CXR/reports/files'
):
    report_path = os.path.join(mimic_reports_path,
                                f'p{subject_id}'[:3],
                                f'p{subject_id}',
                                f's{study_id}.txt')    
    with open(report_path, 'r') as report:
        txt = report.read()
    print(txt)

def show_jpg_and_report(
    subject_id = None,
    study_id = None,
    mimic_images_path = '/media/wonjun/New Volume1/MIMIC-CXR/MIMIC_CXR/files',
    mimic_reports_path = '/media/wonjun/New Volume1/MIMIC-CXR/MIMIC_CXR/reports/files'
):
    show_jpgs(subject_id, study_id, mimic_images_path)
    show_report(subject_id, study_id, mimic_reports_path)



#%%
p = './mimic-cardiac-device-labels/mimic-cxr-cardiac-device-labels3.csv'
df = pd.read_csv(p)
df = df.fillna(0)
df
# %% CXRs with both Edema and Cardiac Device
display(df[(df['Edema']==1) & (df['Cardiac Devices']==0)])
#%%
show_jpg_and_report(subject_id=10002428, study_id=50292543)
# %%
