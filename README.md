# Traffic Signs Detector

To use this repository you will need to follow the requirements
listed [here](https://github.com/AlexeyAB/darknet/tree/2fa539779f4e12e264b9e1b2fc463ac7edec165c#requirements) and an 
[anaconda installation of python 3.6](https://www.anaconda.com/distribution/)

To init the project just launch ``init.sh`` script:

```bash
$ chmod +x init.sh
$ ./init.sh
```

Since this script downloads the whole [GTSDB dataset](http://benchmark.ini.rub.de/?section=gtsdb&subsection=news) and initializes [darknet submodule](https://github.com/AlexeyAB/darknet.git), 
it might take a few minutes

* Preparing the data can be performed using **data-preparation.py** script
* Training and running the network can be performed following AlexeyAB indications 

### References:

This repo is using **AlexeyAB** repo and **YoloV3**
