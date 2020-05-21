# Playing Flappy Bird with Deep Reinforcement Learning

# Highlights:
* Supports macOS, Linux and Windows
* Supports GPU and CPU
* Uses PlaidML (https://github.com/plaidml/plaidml)

# Dependencies:
* Python 3.6+
* pip install -r req.txt

# How to Run?

* Train the network
```
python run.py -m train
```
If you need to train again from scratch, delete model.h5 then run the above command.

* Test without run (you need to have a file model.h5)
```
python run.py -m test
```

* For help:
```
python run.py -h
```

# Credits
Modified from https://github.com/yanpanlau/Keras-FlappyBird
