# Age, gender, emotion estimation
## Demo

* Instructions for demo
```sh
python3 demo.py input_video_path="input/dinner.mp4" \
```

## Train
* Instructions for demo
  * If train with backone 'ShuffleNet V2', using train_shufflenet.py
  * If train with backone 'WideResNet', using train.py
```
python3 train_shufflenet.py
--input_agender
data/imdb_db.mat
--input_wiki
data/wiki_db.mat
--input_emotion
/media/vincentkao/Data/datasets/fer2013/fer2013.csv
--nb_epochs
30
--staircase_decay_at_epochs
(5,8,)
--lr
0.1
--validation_split
0.15
--batch_size
64
```

## References
[1]  R. Rothe, R. Timofte, and L. V. Gool, "DEX: Deep EXpectation of apparent age from a single image," in Proc. of ICCV, 2015.
[2]  [yu4u/age-gender-estimation](https://github.com/yu4u/age-gender-estimation)
[3]  [opconty/keras-shufflenetV2](https://github.com/opconty/keras-shufflenetV2)
[4]  [lmeulen/AgeGenderEmotion](https://github.com/lmeulen/AgeGenderEmotion)
