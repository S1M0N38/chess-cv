# Changelog

## [0.5.0](https://github.com/S1M0N38/chess-cv/compare/v0.4.0...v0.5.0) (2025-10-14)


### Features

* add data augmentation for snap model ([0ee3e6b](https://github.com/S1M0N38/chess-cv/commit/0ee3e6b9ce55267e243a0700a0a977afce721ae6))
* add mouse cursor png to data ([b6f981e](https://github.com/S1M0N38/chess-cv/commit/b6f981ea9993bc9062630f307010f790b426d19b))
* add mouse overlay step to augmentation pipeline ([79af1a1](https://github.com/S1M0N38/chess-cv/commit/79af1a1ee4318e28c1bd5b272f7c1f66f613932c))
* add preprocessing for snap model ([3cc638b](https://github.com/S1M0N38/chess-cv/commit/3cc638b76fa8434b0de44b2b43f5efbd9b75548b))
* add snap weights ([82fa484](https://github.com/S1M0N38/chess-cv/commit/82fa48429c56ac71314eefbc6680d8874d5f8491))
* def constants for snap model ([86beac9](https://github.com/S1M0N38/chess-cv/commit/86beac9b07ce00ff765fd19beebb284a1651a154))
* generate more (8) variations per image for snap ([233473d](https://github.com/S1M0N38/chess-cv/commit/233473dd4b940dccbd393a2d8df49e48ffd73130))
* new pieces model weights ([9a0d8f1](https://github.com/S1M0N38/chess-cv/commit/9a0d8f1f6ce67f5c27c7ee5cb79f4109428a5b35))
* update augmentations configs ([bb286b6](https://github.com/S1M0N38/chess-cv/commit/bb286b637ad1566fc248fdec13aad3283f29d4cb))
* update constants for snap model preprocessing ([4179103](https://github.com/S1M0N38/chess-cv/commit/41791031703c464dca1b41e85047eacbd287b401))


### Bug Fixes

* preprocessing for snap model ([0c5546b](https://github.com/S1M0N38/chess-cv/commit/0c5546be6a1a5f61edba5eea53ed933a11a73cde))
* simplify preprocessing for snap model ([8b440a2](https://github.com/S1M0N38/chess-cv/commit/8b440a22d8ab51368db968201c9c5973011ae664))


### Documentation

* add sample images for snap model ([9d3e800](https://github.com/S1M0N38/chess-cv/commit/9d3e80023333ae8ceabc9056a66333d9b9cacd8a))
* add snap model to docs (WIP) ([6c9b53e](https://github.com/S1M0N38/chess-cv/commit/6c9b53ec43b155d141f5c1e69f476480e8e372a5))
* update citations and huggingface readme ([ffc3758](https://github.com/S1M0N38/chess-cv/commit/ffc375845fdf33b28c843a4d4a2b611bd91061c0))
* update data augmentation with mouse overlay ([c2ac0e5](https://github.com/S1M0N38/chess-cv/commit/c2ac0e5bd7b7d2432cab6f34b0954a30dcc56403))
* update metrics for new pieces model ([94f9ba8](https://github.com/S1M0N38/chess-cv/commit/94f9ba8f3851df6f3a42d56915598cb47c4ee354))
* update repo subtitle ([ab210b1](https://github.com/S1M0N38/chess-cv/commit/ab210b112f81218870cb9f844d67d93082fc46be))
* update sample images for the snap model ([f4b0fe9](https://github.com/S1M0N38/chess-cv/commit/f4b0fe9bf7a0d8eb7f4ffdd63cd3224d46df557a))
* update snap dataset size and escape `~` in markdown ([a989fc4](https://github.com/S1M0N38/chess-cv/commit/a989fc4b90f19bc27e45ee72469a3934e2d81bac))
* update snap model samples for data aug ([b31e978](https://github.com/S1M0N38/chess-cv/commit/b31e9783ac242fb6f297f7ad5465a494e12a33aa))
* update snap scores (good-wood-12) ([fe5e0a8](https://github.com/S1M0N38/chess-cv/commit/fe5e0a8836d545071d09423684b5801e199c06ef))
* update the script to generate snap augmentation examples ([0313d2c](https://github.com/S1M0N38/chess-cv/commit/0313d2ca6ba3fa9a8e7911f96247f0bb8e69a452))

## [0.4.0](https://github.com/S1M0N38/chess-cv/compare/v0.3.0...v0.4.0) (2025-10-12)


### Features

* add data augmentations page to mkdocs ([0647f3d](https://github.com/S1M0N38/chess-cv/commit/0647f3d7d500b1e5f92c87b28b912037ace65b4f))
* add lr-scheduler ([bc83e03](https://github.com/S1M0N38/chess-cv/commit/bc83e03146b7fc93c9940f7ba869a77604f7e38f))
* add model weights to src ([33c52f1](https://github.com/S1M0N38/chess-cv/commit/33c52f1b56ff77c91fb4927dac016153340511db))
* add transparency to lichess arrows and highlights ([e070f5e](https://github.com/S1M0N38/chess-cv/commit/e070f5ed6f0ac6de421ca9513beb6de767a1808c))
* update __init__.py ([b0b610c](https://github.com/S1M0N38/chess-cv/commit/b0b610ca25ecfbbeb7f82dee5dfc620856ba68a2))
* update data augmentations parameters for model pieces ([9c9b9eb](https://github.com/S1M0N38/chess-cv/commit/9c9b9eb51c26b6a47d7022e897e5545c5989818d))
* update data augmentations pipelines ([36eaecd](https://github.com/S1M0N38/chess-cv/commit/36eaecd8dbf4377c5c7ce1bf72a4701e39d876e2))
* use alpha composition for overlays ([15d7402](https://github.com/S1M0N38/chess-cv/commit/15d7402cac5ceefb3059cd7652bafa4431582298))


### Bug Fixes

* make validate epoch pbar leave=False ([7396c48](https://github.com/S1M0N38/chess-cv/commit/7396c485a89951f5de210204f14e15b10f616c59))
* reduce brightness jitter in augmentations for pieces ([1fa4885](https://github.com/S1M0N38/chess-cv/commit/1fa4885b14876d33d7049bd836227703cb33fcd7))
* remove print statement that was messing up progress bar ([645c0ac](https://github.com/S1M0N38/chess-cv/commit/645c0ac2dde9b66a52cd480fb0c3f5172ec28542))


### Documentation

* update `update-metrics.md` command ([7500def](https://github.com/S1M0N38/chess-cv/commit/7500def717947f9479d56ac28a1187e2b09cffba))
* update data augmentations docs ([abe0caf](https://github.com/S1M0N38/chess-cv/commit/abe0caf3ce024706303dfd082c8e9fc59b5cd6b5))
* update hyper-parameters for training pieces model ([d95078a](https://github.com/S1M0N38/chess-cv/commit/d95078ac6ec69a615db358ebcd6d523c7032dd88))
* update metrics with new pieces model ([24edbb4](https://github.com/S1M0N38/chess-cv/commit/24edbb4ca88bcaeba9dba05a34c3c440eb6271e8))
* update metrics with new pieces model ([5427389](https://github.com/S1M0N38/chess-cv/commit/5427389cea08c45b2564a3e0a5002e250da4f9c6))
* update setup instructions ([7889d8a](https://github.com/S1M0N38/chess-cv/commit/7889d8a746d43b920a9f5d14b61f67e7608a1559))
* update the inference to use bundled models ([b6f753f](https://github.com/S1M0N38/chess-cv/commit/b6f753fe07555927d954cb0c17cb433a32c6662a))

## [0.3.0](https://github.com/S1M0N38/chess-cv/compare/v0.2.1...v0.3.0) (2025-10-09)


### Features

* add arrows and highlights in data ([3c31bb5](https://github.com/S1M0N38/chess-cv/commit/3c31bb575c90e84c4e1d838d2d4d03fa6b1d5fc6))
* add arrows model to eval target ([1273f7f](https://github.com/S1M0N38/chess-cv/commit/1273f7f7b6f390a082602ce46d07a7b74ab99ca8))
* add augmentation configurations ([3471120](https://github.com/S1M0N38/chess-cv/commit/347112033d4510c19f075f760fb5ca20c189721c))
* add hashed subdirs to preprocessing ([6e505cd](https://github.com/S1M0N38/chess-cv/commit/6e505cd409c337386d619cfe6b9417916d7390a1))
* add inference speed benchmarking ([d002651](https://github.com/S1M0N38/chess-cv/commit/d002651976db3ca12a7bb17bc94df17ad6f94209))
* add logging mid-epoch ([019de5e](https://github.com/S1M0N38/chess-cv/commit/019de5e777388e89eb0f496bc92ea35bc3abae3e))
* add multi-split dataset concatenation support ([85a2093](https://github.com/S1M0N38/chess-cv/commit/85a209333a4118969c2914880d7cb2d9de4c21b7))
* add preprocessing for arrows model ([3944b7d](https://github.com/S1M0N38/chess-cv/commit/3944b7dfe49c8ecdea485a1de27b94f1bc78a60a))
* add proper CLI for chess-cv ([b6489c2](https://github.com/S1M0N38/chess-cv/commit/b6489c2d069b4d24df10ed7bb9ee6508c6bb0074))
* add run_cap to sweep.yaml ([e48deec](https://github.com/S1M0N38/chess-cv/commit/e48deeca1ebd3d4e2ef7e6815e8246b08eff2ca7))
* add val logging mid-epoch ([c3dfec5](https://github.com/S1M0N38/chess-cv/commit/c3dfec5d0dd3c31c32d73933f6983ef56c19a3f9))
* add W&B hyperparameter sweep functionality ([70e64f8](https://github.com/S1M0N38/chess-cv/commit/70e64f837c8756644fb82c663eb6a89cea3d8366))
* adjust augmentations for arrows model ([497a6d7](https://github.com/S1M0N38/chess-cv/commit/497a6d7bc131f5ac010003f99faf4d689ae05560))
* enhance test logging with W&B tables and artifacts ([613c094](https://github.com/S1M0N38/chess-cv/commit/613c0943a01e05f9f3bd1fab04da6ff92c1de6c3))
* enhance W&B logging capabilities ([49e0814](https://github.com/S1M0N38/chess-cv/commit/49e081475777f6d9602772a25e3f1bfa60622938))
* update metrics for pieces model ([285fbbd](https://github.com/S1M0N38/chess-cv/commit/285fbbd55e23e8baefb4213633012935cd94769f))
* update the data augmentations to include arrow and highlight overlays ([1bfcb9a](https://github.com/S1M0N38/chess-cv/commit/1bfcb9a6eabcaf9fd4c2385de81a7fd23ea70d8d))


### Bug Fixes

* add defensive null check for wandb_logger ([a03c1fe](https://github.com/S1M0N38/chess-cv/commit/a03c1fe7ae99c47e62ef741b16dd43d28bf3e24d))
* add no-arrows image to preprocessing arrows ([8f069b3](https://github.com/S1M0N38/chess-cv/commit/8f069b3ce0a4bbef2ae1906e89edb660c26f7cc1))
* add piece class to arrow image name ([c5670d8](https://github.com/S1M0N38/chess-cv/commit/c5670d8f6e0e67c76c842cc8d75d8ac34a4bd5a5))
* correct label extraction for hashed subdirectories ([37c9bfd](https://github.com/S1M0N38/chess-cv/commit/37c9bfdef5aa761636a936a5c9658595d8c06d89))
* limit number of misclassified images to save ([f327b20](https://github.com/S1M0N38/chess-cv/commit/f327b20532807cbbf84a55188be1af8b2e5c2894))
* make data augmentation more robust for arrow images ([80f2425](https://github.com/S1M0N38/chess-cv/commit/80f24259e2cfeceb883952e33ef2a04a709decd8))
* make progress bar smoother in preprocessing ([ad796b1](https://github.com/S1M0N38/chess-cv/commit/ad796b1daa90ba6663d6760bb7bf8a0c78a007c2))
* preprocessing for arrows model ([3ae0384](https://github.com/S1M0N38/chess-cv/commit/3ae03844463d07f79918a1f8471a2505419ed1df))
* reduce the number of epochs for arrows model ([319f7eb](https://github.com/S1M0N38/chess-cv/commit/319f7ebd88deb197ff02ccf22182e528b675b0ef))
* remove flags from CLI to make less bloated ([78d6c3b](https://github.com/S1M0N38/chess-cv/commit/78d6c3b47f946f0a6933fcb0b43b41868952e630))
* remove support for hashed subdirectories ([dfa2de8](https://github.com/S1M0N38/chess-cv/commit/dfa2de8e2756ae93eae1662bfcd849fd50a6dfdb))
* udpate arrows image with proper rotation (avoid EXIF) ([c592429](https://github.com/S1M0N38/chess-cv/commit/c592429baae117f076d982436536fa6b030d09e0))
* update batch size for arrows sweep ([bb1e707](https://github.com/S1M0N38/chess-cv/commit/bb1e707439d0f33f4312dd3081594bc16387b4bc))
* use {model_id}.safetensors instead of best_model.safetensors ([b281111](https://github.com/S1M0N38/chess-cv/commit/b28111130faa0ed9891d88f52faa006919a56985))
* use BEST_MODEL_FILENAME constant in upload ([9fa346b](https://github.com/S1M0N38/chess-cv/commit/9fa346bf4c908bfed4a720fd8aeecb9a37379580))


### Documentation

* add ChessVision evaluation metrics ([e008667](https://github.com/S1M0N38/chess-cv/commit/e0086678e26b4ea36ec744c75c0bba6e2868b630))
* add inference speed to architecture ([82179b1](https://github.com/S1M0N38/chess-cv/commit/82179b138cffa2abce292dfbe7c2260cc5d07bf7))
* add llms.txt badge ([fa0cf92](https://github.com/S1M0N38/chess-cv/commit/fa0cf92649bd03df642ea2464cba32d865dd148e))
* add models section with training details and performance metrics ([67c93b6](https://github.com/S1M0N38/chess-cv/commit/67c93b673643c1136619c8b7a8f5f9b8563415aa))
* **metrics:** update evaluation results across documentation ([c0ca3d8](https://github.com/S1M0N38/chess-cv/commit/c0ca3d8d1ec010ef030f50c56faa1a8b65d95b55))
* update architecture with arrows model ([e230df4](https://github.com/S1M0N38/chess-cv/commit/e230df4ed9f4cf497c6c1fb1038faec2a69bc1a2))
* update checkpoint references to model-specific naming ([fb410fc](https://github.com/S1M0N38/chess-cv/commit/fb410fc9890604b68b00c92c01dd70e65c4d75c8))
* update model performance metrics ([f3e5323](https://github.com/S1M0N38/chess-cv/commit/f3e5323f7bfe4378dd37139665482c96d166e476))
* update readme with new arrows model ([1590941](https://github.com/S1M0N38/chess-cv/commit/1590941c553b1c5faa5ab343aadad5903d2edd68))
* update README_hf.md citation ([0e7cb2d](https://github.com/S1M0N38/chess-cv/commit/0e7cb2d3e6463cd67d69c818ef10ad4dd9dd0205))
* update README_hf.md with new arrows model ([a44efa7](https://github.com/S1M0N38/chess-cv/commit/a44efa774fe8683d06078ff08fa0e1544ab7fad9))

## [0.2.1](https://github.com/S1M0N38/chess-cv/compare/v0.2.0...v0.2.1) (2025-10-06)


### Documentation

* add citation file ([b45f940](https://github.com/S1M0N38/chess-cv/commit/b45f940376f4e64578031c9828f5d5e27d11fad2))
* centering table in READMEs ([a169796](https://github.com/S1M0N38/chess-cv/commit/a169796e6b815e0f3db30cecc84ca8b7d0120b9b))

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
