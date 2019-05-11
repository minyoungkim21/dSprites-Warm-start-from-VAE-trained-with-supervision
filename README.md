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
- Encoder: Maximize \sum_{(x,z)} \log q(z|x) with sigma of q(z|x) fixed to small value (eg, (1e-4)/3). Ie, only learn the mean function


### 2) Now, standard (unsupervised) VAE learning with initial model from 1)

- Prior p(z) fixed as follows: (z1, z2, z3, z4, z5)
![p_den_2](https://user-images.githubusercontent.com/44901665/57574468-8976e200-7431-11e9-886c-71b9f98df049.jpg)

- sigma of q(z|x) also fixed to ((1e-4)/3). Ie, only learn the mean function.


#### q(z|x) 

- Before learning starts
![qx_den_2](https://user-images.githubusercontent.com/44901665/57574498-f8ecd180-7431-11e9-8ba3-ea06d063a113.jpg)

- At iter# 10
![qx_den_10](https://user-images.githubusercontent.com/44901665/57574518-5f71ef80-7432-11e9-9d52-7f610c423bab.jpg)

- At iter# 20
![qx_den_20](https://user-images.githubusercontent.com/44901665/57574560-33a33980-7433-11e9-8b3e-9c21f160a570.jpg)

- At iter# 35
![qx_den_35](https://user-images.githubusercontent.com/44901665/57574562-3b62de00-7433-11e9-9fd0-cb06c5ba7ed0.jpg)

#### q(z)

- Before learning starts
![q_den_2](https://user-images.githubusercontent.com/44901665/57574489-d490f500-7431-11e9-8daa-9d87a301cdac.jpg)

- At iter# 10
![q_den_10](https://user-images.githubusercontent.com/44901665/57574520-639e0d00-7432-11e9-8cdd-164f2521b152.jpg)

- At iter# 20
![q_den_20](https://user-images.githubusercontent.com/44901665/57574549-21c19680-7433-11e9-95ec-84a75e0f1b89.jpg)

- At iter# 35
![q_den_35](https://user-images.githubusercontent.com/44901665/57574552-271ee100-7433-11e9-9f04-216c0abe83a2.jpg)


#### Latent traversal

- Before learning starts
![fixed0](https://user-images.githubusercontent.com/44901665/57574579-7fee7980-7433-11e9-909e-c2638d020108.gif)

- At iter# 35
![fixed0](https://user-images.githubusercontent.com/44901665/57574583-8aa90e80-7433-11e9-9d95-8a7cd7713134.gif)


