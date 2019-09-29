#!/bin/bash
echo "Initializing the project"
wget https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip
wget https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TestIJCNN2013.zip
unzip TrainIJCNN2013.zip -d .
unzip TestIJCNN2013.zip -d .
conda install -c conda-forge tqdm
git submodule init
git submodule update

wget https://pjreddie.com/media/files/darknet53.conv.74 -P darknet/build/darknet/x64

python ./traffic-detection.py --names_path=data/obj.names  --desc_path=TrainIJCNN2013/ReadMe.txt --in_img_path=TrainIJCNN2013 --out_img_path=data/obj --label_path=TrainIJCNN2013/gt.txt --list_file_path=data/train.txt

mv TestIJCNN2013Download TestIJCNN2013
