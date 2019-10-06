#!/bin/bash
echo "Initializing the traffic detection project"
wget https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip
unzip FullIJCNN2013.zip -d .
conda install -c conda-forge tqdm
git submodule init
git submodule update
wget https://pjreddie.com/media/files/darknet53.conv.74 -P darknet/build/darknet/x64
python ./traffic-detection.py --names_path=data/obj.names  --desc_path=FullIJCNN2013/ReadMe.txt --in_img_path=FullIJCNN2013 --out_img_path=data/obj --label_path=FullIJCNN2013/gt.txt --list_file_dir=data
