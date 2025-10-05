# Changelog

## [0.2.0](https://github.com/S1M0N38/chess-cv/compare/v0.1.1...v0.2.0) (2025-10-05)


### Features

* add evaluation targets ([ffa5e81](https://github.com/S1M0N38/chess-cv/commit/ffa5e813679a0e5b2f2ca71b45196d42506b4ebb))
* add f1 score in test ([23a2853](https://github.com/S1M0N38/chess-cv/commit/23a2853c8a36f91b302d6d04d073370f8105f710))
* add support for huggingface datasets ([36842b5](https://github.com/S1M0N38/chess-cv/commit/36842b5c73293e40f06be04959d2d3ebcc278fdc))
* generate the confusion matrix only locally ([225b691](https://github.com/S1M0N38/chess-cv/commit/225b691bd13815a1f57b7b00814b8b8054ee9822))


### Bug Fixes

* computation of f1 score ([247205d](https://github.com/S1M0N38/chess-cv/commit/247205d23674afa17e7fe9be477d85c4a3b7131b))
* typechecking errors ([c5424ac](https://github.com/S1M0N38/chess-cv/commit/c5424ac209c1b736e931ce27cd048c2c5ba90aec))


### Documentation

* add result tables in the READMEs ([e1a91d3](https://github.com/S1M0N38/chess-cv/commit/e1a91d31a092f76511444bd0ad24e4f06b2c6c28))
* update performance in docs files ([b87c660](https://github.com/S1M0N38/chess-cv/commit/b87c660f2fd2747f50de9d72053fcaef6fa880bf))

## [0.1.1](https://github.com/S1M0N38/chess-cv/compare/v0.1.0...v0.1.1) (2025-10-05)


### Documentation

* make the image bigger in docs home ([1427ff3](https://github.com/S1M0N38/chess-cv/commit/1427ff306f8d54dc9d991507d988d4ca4f8d54d4))
* refine docs ([9ac5fce](https://github.com/S1M0N38/chess-cv/commit/9ac5fce68d353ea607594c5f7c907dd58930cbae))
* update readme section titles ([d5dca66](https://github.com/S1M0N38/chess-cv/commit/d5dca66c27bc8cef45a74e32f28c00902362f527))

## 0.1.0 (2025-10-05)


### Features

* add config file for docs ([3da2d73](https://github.com/S1M0N38/chess-cv/commit/3da2d7313333af4df4613825199c138f29e9d7ce))
* add huggingface readme to docs ([ed35998](https://github.com/S1M0N38/chess-cv/commit/ed35998ec314a29b7dd9b408bece816371cc21f4))
* add huggingface upload script ([bdd552d](https://github.com/S1M0N38/chess-cv/commit/bdd552d26fccb5059489a40d93687f4163e4a0f8))
* add main entry point ([1f34604](https://github.com/S1M0N38/chess-cv/commit/1f34604e7962a8dc6d5751b3f4d19b3e301c32ea))
* add Makefile for development ([f80fd25](https://github.com/S1M0N38/chess-cv/commit/f80fd255d5cd89c293ac26c47396eba03fa32d8a))
* add sweep config ([d52cac7](https://github.com/S1M0N38/chess-cv/commit/d52cac7e92654915be86aba181e3b6b1d4d1fd1b))
* add wandb logging ([a4ef4cf](https://github.com/S1M0N38/chess-cv/commit/a4ef4cf24ce4a317caf2fa4bd6c481ad50ca4596))
* disable early stopping ([f175257](https://github.com/S1M0N38/chess-cv/commit/f17525790488e6f0696048e5340d264bc6e0aefd))
* initial version of the training/evaluation pipeline ([5e0dc39](https://github.com/S1M0N38/chess-cv/commit/5e0dc3953ef3eaa87f1b5fd65adaf0407ffa0841))
* initialize project from template ([c8a9f53](https://github.com/S1M0N38/chess-cv/commit/c8a9f53d233fbc515f9b30c26b53e9ef4ccfe8be))
* new version with data augmentation and smaller model ([e31c603](https://github.com/S1M0N38/chess-cv/commit/e31c603e9d9780113d4e0901706720467841e526))
* update constants ([1c4b6bd](https://github.com/S1M0N38/chess-cv/commit/1c4b6bd3e068bce95b85abc9d87a6e83d9599bc4))
* update default constants ([be30e23](https://github.com/S1M0N38/chess-cv/commit/be30e23f6ed17304f9746f01bbb4cf819c6ea9a2))
* update pieces and boards PNGs ([f6db7d0](https://github.com/S1M0N38/chess-cv/commit/f6db7d018af6f4d8a6d54cf8613376bed88c94f3))
* update preprocessing to generate data from board-piece combinations ([1b8b0d6](https://github.com/S1M0N38/chess-cv/commit/1b8b0d63eb50cb9544cf54259b2a861d32427aeb))
* update README.md ([36a7a8d](https://github.com/S1M0N38/chess-cv/commit/36a7a8dc0917aebb8ea27fa3a037bec92e4a6aba))


### Bug Fixes

* makefile lint and formatting ([5585ac8](https://github.com/S1M0N38/chess-cv/commit/5585ac84406096819e3b4a67f9d6ecfa4c76c687))
* test wandb logging ([aace334](https://github.com/S1M0N38/chess-cv/commit/aace33449e78349a48444750becf1745abbd6ae8))


### Documentation

* add full docs for the project ([690d94b](https://github.com/S1M0N38/chess-cv/commit/690d94b368c7365ecfe0b5266420f327e5cb7fbd))
* add model diagram ([ae24ca4](https://github.com/S1M0N38/chess-cv/commit/ae24ca428fa4bcd399bfec36988b284994c49688))
* add section on W&B ([4064868](https://github.com/S1M0N38/chess-cv/commit/406486871213b03ed6cc8d7fc9c1b8711b37af97))
* expand README with comprehensive usage guide ([e64f869](https://github.com/S1M0N38/chess-cv/commit/e64f8696eb6b18d5c9cee6c2c19e040f2a5358db))
* move the model architecture at the top and add parmas count ([0d9fbcb](https://github.com/S1M0N38/chess-cv/commit/0d9fbcb4492597328847ba923d84b3ba075e3bc7))
* update class names ([1d8d10f](https://github.com/S1M0N38/chess-cv/commit/1d8d10f1fafbdc43761440f7b5b1b02e318bafad))
* update README.md ([74bb6e1](https://github.com/S1M0N38/chess-cv/commit/74bb6e1dab32082c26f01435ee7a8cb15dc91d91))
* update README.md ([94c5f0c](https://github.com/S1M0N38/chess-cv/commit/94c5f0cc8589185a900dd90e84f1f300e191c83c))

## 0.1.0 (Initial Release)

### Features

* Initialize chess-cv project from template
* Add MLX, NumPy, and Matplotlib dependencies
* Set up project structure for CNN-based chess piece classification
