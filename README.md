# Playing Flappy Bird with Deep Reinforcement Learning
* Watch the training process created by the code here on youtube: https://www.youtube.com/watch?v=h-JruqMFUnI&t=30775s

## Highlights:
* Supports macOS, Linux and Windows
* Supports GPU and CPU
* Uses PlaidML (https://github.com/plaidml/plaidml)

## Dependencies:
* Python 3.6+
* pip install -r req.txt

## Installation
* install Python 3.7.2 (Python 3.6+ should work just fine)
* create a venv and activiate it
* pip install -r req.txt
* config plaidml by running plaidml-setup
* modify ~/.keras/keras.json and ~/.plaidml by referring to the examples in setup-plaidml
* done

## How to Run?

* Train the network
```
python train.py
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
Modified from https://github.com/yanpanlau/Keras-FlappyBird


