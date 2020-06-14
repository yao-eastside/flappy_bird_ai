# Playing Flappy Bird with Deep Reinforcement Learning
* Watch the training process created by the code here on youtube: https://www.youtube.com/watch?v=h-JruqMFUnI

## Highlights:
* Supports macOS, Linux and Windows
* Supports GPU and CPU
* Uses PlaidML (https://github.com/plaidml/plaidml)

## Dependencies:
* Python 3.6+
* pip install -r req.txt

## Installation
* You have a Mac (other machines should work as well)
* install Python 3.7.2 (Python 3.6+ should work just fine)
* verify your python version: python3 --version
* create a venv and activiate it: python3 -m venv .venv; source .venv/bin/activate
* pip install -r req.txt, ignore errors/warnings related to tensorflow
* config plaidml by running plaidml-setup. Choose no for experimental in step 1; choose your graphics card in step 2. (https://github.com/zhaoshaojun/flappy_bird_ai/blob/master/plaidml-setup/.plaidml)
* config your backend to be plaidml by modifing ~/.keras/keras.json (https://github.com/zhaoshaojun/flappy_bird_ai/blob/master/plaidml-setup/keras.json). If you do not have this file in your system, run touch ~/.keras/keras.json first, and copy the content over.
* done

## How to Run?

* Train the network
```
python train.py
```
or
```
./start-train.sh
```

* Test
```
python test.py <model-file-name.h5>
```
The <model-file-name.h5> file was generated in the training stage.

## Learn More about Reinforcement Learning
* https://www.youtube.com/watch?v=lvoHnicueoE
* http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf

## Credits
Modified from the outdated https://github.com/yanpanlau/Keras-FlappyBird


