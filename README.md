# Online-Meta-Adaptive-Control

These codes are based on the Online Meta-Adaptive Control (OMAC) algorithm proposed in the work "Meta-Adaptive Nonlinear Control: Theory and Algorithms".

Please run `experiments.ipynb`, where we provide several instantiations of OMAC in a nonlinear pendulum model with external wind disturbance, gravity mismatch, and unknown damping. For the 6-DoF drone experiments, please check out the `quadsim` branch (stay tuned). See the following video for the performance of Deep OMAC in quadrotor control with wind disturbance:

https://user-images.githubusercontent.com/26674297/138949373-413974b8-1164-480b-8d10-99ca91cbc5d3.mp4

Paper link: https://arxiv.org/abs/2106.06098

> @article{shi2021meta,
  title={Meta-Adaptive Nonlinear Control: Theory and Algorithms}, 
  author={Shi, Guanya and Azizzadenesheli, Kamyar and O'Connell, Michael and Chung, Soon-Jo and Yue, Yisong},
  journal={Advances in Neural Information Processing Systems},
  year={2021}}
