# SE-ResUnet in StructSeg2019
## Train and val
To train your model, just set some hyperparameter and run train_2d.py in both coarse and fined segmentation stage. <br>
If you want to train a coarse segmentation model for all organs, set organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Esophagus', 'Trachea', 'SpinalCord']. For fine segmentation of each organ, set organs = ['Bg', 'Trachea'](Trachea for example). <br>
crop_h and crop_w should be 512 for coarse stage, 256 in RightLung, LeftLung, Heart fined segmentation and 128 in Esophagus, Trachea, SpinalCord's due to the different area of different organs in CT slices. <br>
This code only support GPU train and test, set the 'cuda' of your GPU ids used. Other parameter is trvial.<br>

## Data
Arange your data like the text file in data_split folder, train.txt for coarse segmenation and subfolder files for fined segmenation. Use absolute path works well. <br>
For dataset configuration, self.to_tensor and self.seq need to be set for different task, if needed to change, we write as annotatation in the corresponding position.<br>

## Test
To test your model, run val.py file. Set the same hyperparameters used in train.py and set the pretrained model path. Firstly run main, croped2full_pred and evaluate function to get a nii and csv output. Secondly, you can run ensemble function to ensemble mutiple models to get a better result.

## Pretrained models
As the model file is too large to upload, we upload to BaiduYun disc
BaiduYun disc：https://pan.baidu.com/s/1Ux-g0YVDvrTmE7bMLT9VJw 
password：a0o7
It contains all used model in StructSeg2019, 5 fold models of both coarse segmentation stage and all organs segmentation of fined stage.
