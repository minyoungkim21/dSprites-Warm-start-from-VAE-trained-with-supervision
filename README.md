# dSprites: Warm-start from the VAE model trained with full supervision

### 0) Overall setup (brief)

- We first train VAE with full factor labels
- Then, do standard (unsupervised) VAE learning (ie, ELBO) starting from the sup-trained VAE model


### 1) Train VAE with full supervision

#### Fully labeled factors (scaled to [-1,1])
- z1 (shape; card=3) = {-0.9, 0, 0.9}
- z2 (size; card=6) = {-0.9, -0.54, -0.18, 0.18, 0.54, 0.9}
- z3 (rotation; card=40) = {-0.9, -0.88, ..., 0.88, 0.9}
- z4 (x-pos, card=32) = {-0.9, -0.87, ..., 0.87, 0.9}
- z5 (y-pos, card=32) = {-0.9, -0.87, ..., 0.87, 0.9}

#### Learning p(x|z) and q(z|x) with observed (x,z)
- Decoder: Maximize \sum_{(x,z)} \log p(x|z) 
- Encoder: Maximize \sum_{(x,z)} \log q(z|x) with sigma of q(z|x) fixed to small value (eg, (1e-4)/3). Ie, only learn the mean function


### 2) Now, standard (unsupervised) VAE learning with initial model from 1)

- We fixed prior p(z) as follows: (z1, z2, z3, z4, z5)

![p_den_2](https://user-images.githubusercontent.com/44901665/57574468-8976e200-7431-11e9-886c-71b9f98df049.jpg)

- sigma of q(z|x) also fixed to ((1e-4)/3). Ie, only learn the mean function.


#### q(z)

- Before learning starts (ie, the supervised-trained VAE)

![q_den_2](https://user-images.githubusercontent.com/44901665/57574489-d490f500-7431-11e9-8daa-9d87a301cdac.jpg)

- At iter# 35

![q_den_35](https://user-images.githubusercontent.com/44901665/57574552-271ee100-7433-11e9-9f04-216c0abe83a2.jpg)

- At iter# 100

![q_den_100](https://user-images.githubusercontent.com/44901665/57580425-5ec47200-74a1-11e9-89bd-f03361864416.jpg)

- At iter# 200

![q_den_200](https://user-images.githubusercontent.com/44901665/57580426-5ec47200-74a1-11e9-801c-6887332ca603.jpg)

- At iter# 300

![q_den_300](https://user-images.githubusercontent.com/44901665/57580427-5ec47200-74a1-11e9-886d-d78692eedb82.jpg)


#### q(z|x) 

- Before learning starts

![qx_den_2](https://user-images.githubusercontent.com/44901665/57574498-f8ecd180-7431-11e9-8ba3-ea06d063a113.jpg)

- At iter# 35

![qx_den_35](https://user-images.githubusercontent.com/44901665/57574562-3b62de00-7433-11e9-9fd0-cb06c5ba7ed0.jpg)

- At iter# 100

![qx_den_100](https://user-images.githubusercontent.com/44901665/57580448-9f23f000-74a1-11e9-96a8-3ef531f9f4b2.jpg)

- At iter# 200

![qx_den_200](https://user-images.githubusercontent.com/44901665/57580449-9f23f000-74a1-11e9-926c-9bd5b3eec475.jpg)

- At iter# 300

![qx_den_300](https://user-images.githubusercontent.com/44901665/57580450-9fbc8680-74a1-11e9-9e3e-cfba5e7c34df.jpg)


#### Latent traversal

- Before learning starts

![fixed0](https://user-images.githubusercontent.com/44901665/57574579-7fee7980-7433-11e9-909e-c2638d020108.gif)

- At iter# 35

![fixed0](https://user-images.githubusercontent.com/44901665/57574583-8aa90e80-7433-11e9-9d95-8a7cd7713134.gif)


#### Loss functions and evaluation metrics

- recon loss

![recon_loss_new](https://user-images.githubusercontent.com/44901665/57580405-09886080-74a1-11e9-8b7d-d677fd4236f0.png)

- kl loss

![kl_loss_new](https://user-images.githubusercontent.com/44901665/57580403-08efca00-74a1-11e9-9e02-88f73f1778ab.png)

- metrics

![metrics_new](https://user-images.githubusercontent.com/44901665/57580404-08efca00-74a1-11e9-9597-3062a32e3c48.png)




### 3) Conclusion

- The unsupervised VAE learning doesn't change anything (loss, q(z|x), q(z)) significantly.
- So, the VAE model trained with full labels seems to be a local minima with respect to the VAE loss.
- A peculiar thing is the reconstruction loss is quite large (compared to VAE learning with a random initial model, which converges to recon loss <50). Is it because we fixed the sigma of q(z|x) (as small value (1e-4)/3)?

