**UGRec:<br />**
This is a simple algorithm implementation for the paper [UGRec: Modeling Directed and Undirected Relations for
Recommendation](https://doi.org/10.1145/3404835.3462835) in the Conference SIGIR'21.<br />
**Environment Settings:**<br />
Python 3.8 + Tensorflow-gpu version 2.1.0 <br />
**Example to run the codes:**<br />
Run ugrec.py <br />
python ugrec.py --dataset game --embed_size 64 --lr 0.05 --batch_size 200 --epoch 1000 --Ks 20 --margins [1.6,1.0,0.9]