# Defect_dP_PaCKage
## Machine learning on defect identification

Some demo you could find:
https://youtu.be/XkCfma-XxoA 


<img src="https://images.zapnito.com/cdn-cgi/image/metadata=copyright,fit=scale-down,format=auto,sharpen=1,quality=95/https://images.zapnito.com/users/570443/posters/1650814533-64-0151/8aea5112-c4e0-44c0-94a5-a897ac38a28c_large.jpeg" alt="Data Image">


**Some packages for PCA analysis so that you can practice.

*** Active learning includes deep kernel learning. 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziatdinovmax/gpax/blob/main/examples/gpax_viDKL_plasmons.ipynb)

[Deep kernel learning (DKL)](https://arxiv.org/abs/1511.02222) can be understood as a hybrid of deep neural network (DNN) and GP. The DNN serves as a feature extractor that allows reducing the complex high-dimensional features to low-dimensional descriptors on which a standard GP kernel operates. The parameters of DNN and of GP kernel are inferred jointly in an end-to-end fashion. Practically, the DKL training inputs are usually patches from an (easy-to-acquire) structural image, and training targets represent a physical property of interest derived from the (hard-to-acquire) spectra measured in those patches. The DKL output on the new inputs (image patches for which there are no measured spectra) is the expected property value and associated uncertainty, which can be used to derive the next measurement point in the automated experiment. 

GPax package has the fully Bayesian DKL (weights of neural network and GP hyperparameters are inferred using Hamiltonian Monte Carlo) and the Variational Inference approximation of DKL, viDKL. The fully Bayesian DKL can provide an asymptotically exact solution but is too slow for most automated experiments. Hence, for the latter, one may use the viDKL
```python3
import gpax

# Get random number generator keys for training and prediction
rng_key, rng_key_predict = gpax.utils.get_keys()

# Obtain/update DKL posterior; input data dimensions are (n, h*w*c)
dkl = gpax.viDKL(input_dim=X.shape[-1], z_dim=2, kernel='RBF')  # A
dkl.fit(rng_key, X_train, y_train, num_steps=100, step_size=0.05)  # B

# Compute UCB acquisition function
obj = gpax.acquisition.UCB(rng_key_predict, dkl, X_unmeasured, maximize=True)  # C
# Select next point to measure (assuming grid data)
next_point_idx = obj.argmax()  # D

# Perform measurement in next_point_idx, update measured & unmeasured data arrays, and re-run steps A-D.
```
Below we show a result of a simple DKL-based search for regions of the nano-plasmonic array that [host edge plasmons](https://arxiv.org/abs/2108.03290). The full example is available [here](https://colab.research.google.com/github/ziatdinovmax/gpax/blob/main/examples/gpax_viDKL_plasmons.ipynb). 

<img src="https://user-images.githubusercontent.com/34245227/160270568-147fa21b-91f3-48b8-8dd2-c33eb4b497b4.png">

Note that in viDKL, we use a simple MLP as a default feature extractor. However, you can easily write a custom DNN using [haiku](https://github.com/deepmind/dm-haiku) and pass it to the viDKL initializer
```python3
import haiku as hk

class ConvNet(hk.Module):
    def __init__(self, embedim=2):
        super().__init__()
        self._embedim = embedim   

    def __call__(self, x):
        x = hk.Conv2D(32, 3)(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(2, 2, 'SAME')(x)
        x = hk.Conv2D(64, 3)(x)
        x = jax.nn.relu(x)
        x = hk.Flatten()(x)
        x = hk.Linear(self._embedim)(x)
        return x

dkl = gpax.viDKL(X.shape[1:], 2, kernel='RBF', nn=ConvNet)  # input data dimensions are (n,h,w,c)
dkl.fit(rng_key, X_train, y_train, num_steps=100, step_size=0.05)
obj = gpax.acquisition.UCB(rng_key_predict, dkl, X_unmeasured, maximize=True)
next_point_idx = obj.argmax()
```
## Installation
If you would like to utilize a GPU acceleration, follow these [instructions](https://github.com/google/jax#installation) to install JAX with a GPU support.

Then, install GPax using pip:

```$ pip install git+https://github.com/ziatdinovmax/gpax```

If you are a Windows user, we recommend to use the Windows Subsystem for Linux (WSL2), which comes free on Windows 10 and 11.


Check our DEMO project
<a href=https://github.com/nicolesbishop/Datascience> HERE. </a>
  
<a href=https://github.com/py4dstem/py4DSTEM> -Py4DSTEM-</a>
