# GlyphReader
A deeplearning approach to classifying the ancient Egyptian hieroglyphs. The source code is written in **python3** using the popular [Keras](https://keras.io/) framework (with a [Tensorflow](https://keras.io/) backend).
It attempts to classify images to their [Gardiner](https://en.wikipedia.org/wiki/Gardiner's_sign_list) labels, such as:

**Image** | ![GitHub Logo](/examples/200000_S29.png) | ![GitHub Logo](/examples/200001_V13.png) | ![GitHub Logo](/examples/200003_G43.png) 
------------ | ------------ | ------------- | -------------
**Gardener Label** | S29 | V13 | G43

In addition to the source code, we also provide a dataset containing 4210 manually annotated images of Egyptian hieroglyphs found in the [Pyramid of Unas](https://en.wikipedia.org/wiki/Pyramid_of_Unas).
The dataset will be automatically downloaded when using `train.py` to train a new classifier, but is also available [here](http://iamai.nl/downloads/GlyphDataset.zip)

## Requirements
- `pip3 install numpy sklearn scipy pyyaml h5py`
- [tensorflow](https://www.tensorflow.org/get_started/os_setup)
- [keras](https://keras.io/#installation)

## Usage
```
python3 src/classify.py examples
```
Expected output:
```
Predicting the Hieroglyph type...
image name                ::: top 5 best matching hieroglyphs
200000_S29.png            --> ['S29' 'V25' 'V7' 'O4' 'V28']
200001_V13.png            --> ['V13' 'D36' 'N18' 'T21' 'F18']
200002_V13.png            --> ['V13' 'T30' 'U7' 'F18' 'N1']
200003_G43.png            --> ['G43' 'G17' 'G21' 'E1' 'G25']
200004_D21.png            --> ['D21' 'N19' 'V30' 'G43' 'X6']
200005_O50.png            --> ['O50' 'N5' 'O49' 'W24' 'D1']
200006_X1.png             --> ['X1' 'G43' 'D4' 'N29' 'D21']
200007_M23.png            --> ['M23' 'M26' 'G39' 'U35' 'M12']
200008_G43.png            --> ['G43' 'G21' 'G29' 'M17' 'G1']
200009_S29.png            --> ['S29' 'O4' 'Y3' 'V25' 'M195']
200010_V13.png            --> ['V13' 'T30' 'D52' 'U7' 'D36']
200011_M23.png            --> ['M23' 'M17' 'D56' 'M26' 'U1']
200012_G43.png            --> ['G43' 'G35' 'G21' 'M23' 'G4']
200013_D21.png            --> ['D21' 'O34' 'O29' 'N14' 'N19']
200014_O50.png            --> ['O50' 'G17' 'X1' 'N17' 'D19']
200015_V13.png            --> ['V13' 'D46' 'V31' 'F22' 'N36']
200016_G43.png            --> ['G43' 'G17' 'G1' 'W14' 'G4']
200017_S29.png            --> ['S29' 'O4' 'V28' 'Z7' 'N35']
```

## Training
In case you would like to train your own classifier, use `train.py`. It takes no arguments, but when running it for the first time it will download the dataset, and starts training. Training itself consist of 2 phases:

1. **Feature Extraction** extract deeplearning features from the images (corresponding to the `avg_pool` layer from the `InceptionV3` network).
2. **Train Classifier** train an SVM on the deeplearning features
If you do not have a GPU, or simply want to retrain the classifier, it is possible to skip the first step and download the precomputed features directly at [http://iamai.nl/downloads/features.npy](http://iamai.nl/downloads/features.npy), store them in `intermediates/features.npy`.
