import os
import shutil

import xml.etree.ElementTree as ET
import PIL.Image as Image
import torchvision.transforms as T

def cut_image():
    root_path = "F:/Semi-Supervised/data/20220524dataset_leaf"
    image_path = f"{root_path}/image"
    xml_path = f"{root_path}/xml"
    contents = os.listdir(xml_path)
    
    idx = 1
    for i in contents:
        tree = ET.parse(f'{xml_path}/{i}')
        root = tree.getroot()
        name = ''
        box = []
        source_image = Image.open(f'{image_path}/{i.split(".")[0]}.jpg')

        objs = []
        for r in root.iter():
            if 'name' == r.tag:
                if r.text not in objs:
                    
                    objs.append(r.text)

        for child in root.iter():
            
            if 'name' == child.tag:
                name = child.text

                if len(objs) == 1:
                    save_path = f'{root_path}/train/{name}'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    source_image.save(f'{save_path}/{idx}_{i.split(".")[0]}.jpg')
                    idx += 1
                    break

            elif child.tag in ['xmin','ymin', 'xmax', 'ymax']:
                box.append(int(child.text))
            if len(box) == 4:
                print(name, box)
                img = source_image.crop(tuple(box))
                save_path = f'{root_path}/train/{name}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                name = ''
                box = []

                img.save(f'{save_path}/{idx}_{i.split(".")[0]}.jpg')
                idx += 1            

def count_image():
    root_path = "data/20220524dataset_leaf/leaf_num_not_enough"

    folders = os.listdir(root_path)
    total = 0
    disease_dict = {}
    for folder in folders:
        files = os.listdir(f'{root_path}/{folder}')
        # print(folder, len(files))
        disease_dict[folder] = len(files)
        total += len(files)
    for k, v in sorted(zip(disease_dict.keys(), disease_dict.values()), key=lambda x: x[1], reverse=True):
        print(k)
    print(total)

def resize_image():
    size = 320
    root_path = "F:/Semi-Supervised/data/leaf"
    output_path = f"F:/Semi-Supervised/data/leaf_{size}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    folders = os.listdir(root_path)
    for folder in folders:
        sub_output_folder = f'{output_path}/{folder}'
        if not os.path.exists(sub_output_folder):
            os.makedirs(sub_output_folder)
        for file in os.listdir(f'{root_path}/{folder}'):
            img = Image.open(f'{root_path}/{folder}/{file}')
            w, h = img.size
            pad = abs(w - h)
            if w > h:
                img = T.Pad((0, pad//2, 0, (pad-pad//2)), fill=(114, 114, 114))(img)
            elif h > w:
                img = T.Pad((pad//2, 0, (pad-pad//2), 0), fill=(114, 114, 114))(img)
            
            img = img.resize((size,size))
            img.save(f'{sub_output_folder}/{file}')
            
if __name__ == '__main__':
    count_image()
    # cut_image()