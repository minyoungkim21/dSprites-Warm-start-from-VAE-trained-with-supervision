# dSprites: Warm-start from the VAE model trained with full supervision

### 0) Overall setup (brief)

- Train VAE with full factor labels
- Then, do standard (unsupervised) VAE learning (ie, ELBO) starting from the sup-trained VAE model


### 1) Train VAE with full supervision

#### Fully labeled factors (scaled to [-1,1])
- z1 (shape; card=3) = {-0.9, 0, 0.9}
- z2 (size; card=6) = {-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9}
- z3 (rotation; card=40) = {-0.9, -0.88, ..., 0.88, 0.9}
- z4 (x-pos, card=32) = {-0.9, -0.87, ..., 0.87, 0.9}
- z5 (y-pos, card=32) = {-0.9, -0.87, ..., 0.87, 0.9}

#### Learning p(x|z) and q(z|x) with observed (x,z)
- Decoder: Maximize \sum_{(x,z)} \log p(x|z) 
- Encoder: Maximize \sum_{(x,z)} \log q(z|x) with sigma of q(z|x) fixed to small value (eg, (1e-4)/3)



### 2) Latent traversal

4 examples

  [ x | z1 | z2 | z3 | z4 | z5 ]
