# Semi-Supervised Flows PyTorch
Authors: [Andrei Atanov](https://andrewatanov.github.io/), [Alexandra Volokhova](https://scholar.google.com/citations?user=23LOcyMAAAAJ&hl=en), [Arsenii Ashukha](https://senya-ashukha.github.io/), [Ivan Sosnovik](https://scholar.google.at/citations?user=brUsNccAAAAJ&hl=en), [Dmitry Vetrov](https://scholar.google.ca/citations?user=7HU0UoUAAAAJ&hl=en)

This repo contains code for our INNF workshop paper [Semi-Conditional Normalizing Flows for Semi-Supervised Learning](https://arxiv.org/abs/1905.00505)

__Abstract:__
This paper proposes a semi-conditional normalizing flow model for semi-supervised learning. The model uses both labelled and unlabeled data to learn an explicit model of joint distribution over objects and labels. Semi-conditional architecture of the model allows us to efficiently compute a value and gradients of the marginal likelihood for unlabeled objects. The conditional part of the model is based on a proposed conditional coupling layer. We demonstrate performance of the model for semi-supervised classification problem on different datasets. The model outperforms the baseline approach based on variational auto-encoders on MNIST dataset.

[__Poster__](https://docs.google.com/presentation/d/1wSA6RKG4ko2zI9XuVAsJq0dqd-XTPLOiWlJ17XJ1SoQ/edit?usp=sharing)

# Semi-Supervised MNIST classification

Train a Semi-Conditional Normalizing Flows on MNIST with 100 labeled examples:

`python train-flow-ssl.py --config config.yaml`

You can then find logs at `<where-script-launched>/logs/exman-train-flow-ssl.py/runs/<id-date>`

For the convenience we also provide pretrained weights `pretrained/model.torch`, use `--pretrained` flag for loading.

# Credits

* Credits to https://github.com/ferrine/exman for the exman parser.

# Citation

If you found this code useful please cite our paper

```
@article{atanov2019semi,
  title={Semi-conditional normalizing flows for semi-supervised learning},
  author={Atanov, Andrei and Volokhova, Alexandra and Ashukha, Arsenii and Sosnovik, Ivan and Vetrov, Dmitry},
  journal={arXiv preprint arXiv:1905.00505},
  year={2019}
}
```
