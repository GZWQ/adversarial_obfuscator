# adversarial-obfuscator

Unofficial implementation of the paper [Protecting Visual Secrets using Adversarial Nets](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8014908) 

Here is my experiment results.

## Utility Evaluation

Scene 1 : Trained on **original** images, Test on **original** images

Scene 2 : Trained on **original** images, Test on **generated** images

Scene 3 : Trained on **generated** images, Test on **generated** images



| Scene | Accuracy |
| :---: | :------: |
|   1   |  94.61%  |
|   2   | 87.785%  |
|   3   | 90.643%  |



## Privacy Evaluation

Scene 1 : Trained on **original** images, Test on **generated** images

Scene 2 : Trained on **generated** images, Test on **generated** images



| Scene | Accuracy |
| :---: | :------: |
|   1   |   50%    |
|   2   | 83.537%  |



# Acknowledgments

Inspired by [Erik Linder-Norén](https://github.com/eriklindernoren/Keras-GAN/tree/master/dcgan).
