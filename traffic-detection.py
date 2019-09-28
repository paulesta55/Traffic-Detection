import argparse
import os
import re


from PIL import Image
import pandas as pd
from tqdm import tqdm

#python .\traffic-detection.py --names_path=data\obj.names  --desc_path=FullIJCNN2013\ReadMe.txt --in_img_path=FullIJCNN2013 --out_img_path=data\obj --label_path=FullIJCNN2013\gt.txt

class DataLoader:
    
    def __init__(self,names_path,desc_path,label_path,in_img_path,out_img_path,verbose):
        self.verbose = verbose
        self.names_path = names_path
        self.desc_path = desc_path
        self.in_img_path = in_img_path
        self.out_img_path = out_img_path
        self.label_path = label_path
        self.gtsdb = pd.read_csv(os.path.abspath(label_path), sep=";", header=None,names=
                                 ["img", "x1", "y1", "x2", "y2","id"])
        print(self.gtsdb.head())
        
    def __getitem__(self,idx):
        return {'img':self.load_image(idx),
                'label':self.gtsdb.iloc[idx]['id'],
                'name':self.gtsdb.iloc[idx]['img'],
                'x1':self.gtsdb.iloc[idx]['x1'],
               'x2':self.gtsdb.iloc[idx]['x2'],
               'y1':self.gtsdb.iloc[idx]['y1'],
                'y2':self.gtsdb.iloc[idx]['y2']}
        
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
        
    def dump_all(self):
        for img_name in tqdm(self.gtsdb.img):
            if(self.verbose):
                print(img_name)
            with open(os.path.join(self.out_img_path,os.path.splitext(img_name)[0]+'.txt'),'w') as label_file:
                for index, row in self.gtsdb.loc[self.gtsdb['img'] == img_name].iterrows():
                    bb_coordinates = self.get_bb_coordinates(self[index])
                    
                    if(self.verbose):
                        print(index)
                        print(bb_coordinates)
                    sep = ' '
                    label_file.write(str(bb_coordinates['label'])+sep+str(bb_coordinates['x_center'])+sep+
                                     str(bb_coordinates['y_center'])
                                     +sep+str(bb_coordinates['width'])+sep+str(bb_coordinates['height'])+'\n')
        


def main():
    parser = argparse.ArgumentParser(description="Data loading for traffic sign detection. For instance use "
        +"python .\\traffic-detection.py --names_path=data\\obj.names  --desc_path=FullIJCNN2013\\ReadMe.txt"
 +"--in_img_path=FullIJCNN2013 --out_img_path=data\\obj --label_path=FullIJCNN2013\\gt.txt")
    parser.add_argument('--names_path',type=str,required=True,help="path of the obj.names file required by yolo")
    parser.add_argument('--desc_path',  type=str,required=True,help='path of gtsdb ReadMe.txt file')
    parser.add_argument('--label_path',  type=str,required=True,help='path of the gtsdb gt.txt file')
    parser.add_argument('--in_img_path',  type=str,required=True,help='path of gtsdb ppm images directory')
    parser.add_argument('--out_img_path',  type=str,required=True,help='path of the output directory for images and label files')
    parser.add_argument('--verbose',  type=bool,required=False,help='path of the gtsdb gt.txt file',default=False)
    args = parser.parse_args()
    dataLoader = DataLoader(verbose=args.verbose,names_path=os.path.abspath(args.names_path),
        desc_path=os.path.abspath(args.desc_path),in_img_path=os.path.abspath(args.in_img_path),
        out_img_path= os.path.abspath(args.out_img_path),label_path=os.path.abspath(args.label_path))
    dataLoader.convert_all()
    dataLoader.dump_all()
    dataLoader.dump_classes()


if __name__ == '__main__':
    main()