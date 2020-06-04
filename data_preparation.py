import argparse
import os
import glob
import re

from PIL import Image
import pandas as pd
import cv2 as cv
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#python .\traffic-detection.py --names_path=data\obj.names  --desc_path=FullIJCNN2013\ReadMe.txt --in_img_path=FullIJCNN2013 --out_img_path=data\obj --label_path=FullIJCNN2013\gt.txt --list_file_dir=data
pC = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
mC = [33, 34, 35, 36, 37, 38, 39, 40]
dC = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

ob_names="prohibitory\ndanger\nmandatory\nother"

class DataLoader:
    
    def __init__(self,names_path,desc_path,label_path,in_img_path,out_img_path,verbose,list_file_dir,is_4_cat):
        self.verbose = verbose
        self.names_path = names_path
        self.desc_path = desc_path
        self.in_img_path = in_img_path
        self.out_img_path = out_img_path
        self.label_path = label_path
        self.list_file_dir = list_file_dir
        self.gtsdb = pd.read_csv(os.path.abspath(label_path), sep=";",names=
                                 ["img", "x1", "y1", "x2", "y2","id"])
        self.is_4_cat = is_4_cat
        print(self.gtsdb.dtypes)
        self.complete_dataset()
        if(self.is_4_cat):
            self.class2cat()
        print(self.gtsdb.head())
        
    def complete_dataset(self):
        for file_name in glob.glob(os.path.join(self.in_img_path,'*.ppm')):
            name = os.path.split(file_name)[-1]
            if(name not in self.gtsdb.img.tolist()):
                if(self.verbose):
                    print(str.format("adding {0} to dataset",name))
                image = Image.open(file_name)
                self.gtsdb.loc[len(self)] = [name,-1,-1,-1,-1,-1]
            
            
        
    def __getitem__(self,key):
        if(isinstance(key,slice) ):
            indices = key.indices(len(self))
            return [self[ii] for ii in range(*indices)]
        return {'img':self.load_image(key),
                'label':self.gtsdb.iloc[key]['id'],
                'name':self.gtsdb.iloc[key]['img'],
                'x1':self.gtsdb.iloc[key]['x1'],
               'x2':self.gtsdb.iloc[key]['x2'],
               'y1':self.gtsdb.iloc[key]['y1'],
                'y2':self.gtsdb.iloc[key]['y2']}
        
    def __len__(self):
        return len(self.gtsdb)
        
    
    def load_image(self,idx):
        if(self.verbose):
            print("Loading image at {0}".format(self.gtsdb.iloc[idx]['img']))
        return Image.open(os.path.join(self.in_img_path,self.gtsdb.iloc[idx]['img']))
    
    def format_class_name(self,class_name):
        class_name = class_name.lower().replace(' ','-')+'\n'
        if(self.verbose):
            print(class_name)
        return class_name
    
    def dump_class_name(self,path,class_name):
        with open(os.path.abspath(path),'w') as f:
            f.write(class_name+os.linesep)
            
    def dump_classes(self):
        if(self.is_4_cat):
            with open(os.path.abspath(self.names_path), 'w') as names_file:
                names_file.write(ob_names)
            return
        lines = []
        with open(os.path.abspath(self.desc_path),'r') as desc_file:
            lines = desc_file.readlines()
        if(self.verbose):
            print("".join(lines))
        pattern = re.compile('^\d{1,2}\s=\s(.*)')
        search  = lambda l : re.search(pattern,l)
        names =[match.group(1) for match in map(search,lines) if(match is not None) ]
        with open(os.path.abspath(self.names_path), 'w') as names_file:
            for n in names:
                names_file.write(self.format_class_name(n))
   
    def class2cat(self):
#        class2catDict = {}
#        for c,ids in enumerate([pC,mC,dC]):
#            class2catDict.update({idx:c for idx in ids})    
#       print(class2catDict)
#       self.gtsdb.id = self.gtsdb.apply(lambda x:class2catDict[x.id] if x.id is not np.nan else None,axis=1)   
        for index, row in self.gtsdb.iterrows():
            if(row.id in pC):
                self.gtsdb.loc[index,'id']=0
            elif(row.id in mC):
                self.gtsdb.loc[index,'id']=1
            elif(row.id in dC):
                self.gtsdb.loc[index,'id']=2
            elif(row.id is -1 ):
                self.gtsdb.loc[index,'id']=-1
            else:
                self.gtsdb.loc[index,'id']=3
        
    def convert_all(self):
        cache = []
        for im in tqdm(self):
            if(im['name'] not in cache):
                self.ppm2jpg(im['img'],im['name'])
                cache.append(im['name'])
                    
    
    def ppm2jpg(self,img,img_name):
        os.makedirs(self.out_img_path,exist_ok=True)
        img.save(os.path.join(self.out_img_path,os.path.splitext(img_name)[0]+'.jpg'))
    
    
    def get_bb_coordinates(self,example):
        im_height = example['img'].height
        im_width = example['img'].width
        x_center = (example['x2']+example['x1'])/(2.0*im_width)
        y_center = (example['y2']+example['y1'])/(2.0*im_height)
        height = (example['y2']-example['y1'])/im_height
        width = (example['x2']-example['x1'])/im_width
        return{'label':example['label'],
               'x_center':x_center,
               'y_center':y_center,
               'width':width,
               'height':height}
    

    
    def dump_split(self):
        group_by_img = self.gtsdb.groupby('img')
        count = 0
        train_set = []
        test_set = []
        with open(os.path.join(self.list_file_dir,'train.txt'),'w') as train_file, open(os.path.join(self.list_file_dir,'test.txt'),'w') as test_file: 
            for name,group in tqdm(group_by_img):
                base_name=os.path.splitext(name)[0]
                line = os.path.join(self.out_img_path,os.path.splitext(name)[0]+'.jpg')+'\n'
                if(self.verbose):
                    print(str.format("Appending line {0}",line))
                if(int(base_name)<600):
                    train_file.write(line)
                else:
                    test_file.write(line)
                count += 1
        return train_set,test_set
        
    def dump_all(self):
        for img_name in self.gtsdb.img:
            if(self.verbose):
                print(img_name)
            with open(os.path.join(self.out_img_path,os.path.splitext(img_name)[0]+'.txt'),'w') as label_file:
                for index, row in self.gtsdb.loc[self.gtsdb['img'] == img_name].iterrows():
                    example = self[index]
                    if(example['label'] not in ['-1',-1,None,np.nan]):
                        bb_coordinates = self.get_bb_coordinates(example)
                        if(self.verbose):
                            print(example)
                            print(index)
                            print(bb_coordinates)
                        sep = ' '
                        label_file.write(str(bb_coordinates['label'])+sep+str(bb_coordinates['x_center'])+sep+
                                         str(bb_coordinates['y_center'])
                                         +sep+str(bb_coordinates['width'])+sep+str(bb_coordinates['height'])+'\n')
 
    
                


def main():
    parser = argparse.ArgumentParser(description="Data loading for traffic sign detection. For instance use "
        +"python .\\traffic-detection.py --names_path=data\\obj.names  --desc_path=FullIJCNN2013\\ReadMe.txt"
 +"--in_img_path=FullIJCNN2013 --out_img_path=data\\obj --label_path=FullIJCNN2013\\gt.txt"
 +" --list_file_dir=data")
    parser.add_argument('--names_path',type=str,required=True,help="path of the obj.names file required by yolo")
    parser.add_argument('--desc_path',  type=str,required=True,help='path of gtsdb ReadMe.txt file')
    parser.add_argument('--label_path',  type=str,required=True,help='path of the gtsdb gt.txt file')
    parser.add_argument('--in_img_path',  type=str,required=True,help='path of gtsdb ppm images directory')
    parser.add_argument('--out_img_path',  type=str,required=True,help='path of the output directory for images and label files')
    parser.add_argument('--verbose',  type=bool,required=False,help='path of the gtsdb gt.txt file',default=False)
    parser.add_argument('--list_file_dir',type=str,required=True,help='path of the output file containing the list of the examples')
    parser.add_argument('--is_4_cat',type=bool,required=True,help='specify if you are detecting gtsdb classes or categories(prohibitory,danger,mandatory or other')
    args = parser.parse_args()
    dataLoader = DataLoader(verbose=args.verbose,names_path=os.path.abspath(args.names_path),
        desc_path=os.path.abspath(args.desc_path),in_img_path=os.path.abspath(args.in_img_path),
        out_img_path= os.path.abspath(args.out_img_path),label_path=os.path.abspath(args.label_path),
        list_file_dir=os.path.abspath(args.list_file_dir),is_4_cat=args.is_4_cat)
    dataLoader.convert_all()
    dataLoader.dump_all()
    dataLoader.dump_classes()
    dataLoader.dump_split()



if __name__ == '__main__':
    main()
