# Age, gender, emotion estimation
## Demo

* Instructions for demo
```sh
python3 demo.py input_video_path="input/dinner.mp4" \
```

## Train
* Prepare dataset
  * For age and gender: IMDB + WiKi[6]
  * For emotion: FER-2013[7]
* Instructions for demo
  * If train with backone 'ShuffleNet V2', using train_shufflenet.py
  * If train with backone 'WideResNet', using train.py
  * imdb_db.mat, wiki_db.mat: please refer to [2] to run "create_db.py"
  * fer2013.csv": download from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
```
python3 train_shufflenet.py
--input_agender
data/imdb_db.mat
--input_wiki
data/wiki_db.mat
--input_emotion
data/fer2013.csv
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
[1] R. Rothe, R. Timofte, and L. V. Gool, "DEX: Deep EXpectation of apparent age from a single image," in Proc. of ICCV, 2015.  
[2] [yu4u/age-gender-estimation](https://github.com/yu4u/age-gender-estimation)  
[3] [opconty/keras-shufflenetV2](https://github.com/opconty/keras-shufflenetV2)  
[4] [lmeulen/AgeGenderEmotion](https://github.com/lmeulen/AgeGenderEmotion)  
[5] [oarriaga/face_classification](https://github.com/oarriaga/face_classification)  
[6] [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)  
[7] [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)  
