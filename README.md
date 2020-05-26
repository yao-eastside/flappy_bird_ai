# Playing Flappy Bird with Deep Reinforcement Learning
* Watch the training process created by the code here on youtube: https://www.youtube.com/watch?v=h-JruqMFUnI&t=30775s

## Highlights:
* Supports macOS, Linux and Windows
* Supports GPU and CPU
* Uses PlaidML (https://github.com/plaidml/plaidml)

## Dependencies:
* Python 3.6+
* pip install -r req.txt

## How to Run?

* Train the network
```
python run.py -m train
```
If you need to train again from scratch, delete model.h5 then run the above command.

* Test (you need to have "model.h5" created from training)
```
python run.py -m test
```

* For help:
```
python run.py -h
```

## Learn More about Reinforcement Learning
* https://www.youtube.com/watch?v=lvoHnicueoE
* http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf

## Credits
Modified from https://github.com/yanpanlau/Keras-FlappyBird


