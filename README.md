# NCS-CHALLENGE-SUBMISSION
# TopCoder NCS Challenge Submission: Instructions on how to train the network, export the trained network, compile the network, and how to make inferences on the Movidius NCS using the compiled graph 
https://developer.movidius.com/competition

Code used to train and export the network is in vast majority from the work of [Technica-Corporation/TF-Movidius-Finetune](https://github.com/Technica-Corporation/TF-Movidius-Finetune)

Very minor changes were made to this code, for simplicity it's provided here.

The network implementation and Checkpoints used were taken from the MobileNet TensorFlow slim implementations [models/research/slim/nets/mobilenet_v1.md](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

Added to the NCS Challenge dataset, the ImageNet 2011 Fall Release relevant sysnets were also used [ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.](https://arxiv.org/abs/1409.0575)

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>

## Mobilenet Accuracy Test Results - 80k dataset - Fine Tune all layers.

| Model | Width Multiplier | Image size | Preprocessing | Accuracy-Top1 |Accuracy-Top5 | Log loss | Image Time(ms) | Score |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| MobileNet |1.0 |224 |Same as Inception |79.94% |95.81% |1.89 |41.52 |908890.88 |
| MobileNet |1.0 |192 |Same as Inception |78.97% |95.42% |2.04 |30.12 |905281.95 |
| MobileNet |1.0 |160 |Same as Inception |77.24% |94.61% |2.34 |23.85 |894441.09 |
| MobileNet |0.75 |224 |Same as Inception |76.29% |94.24% |2.49 |26.97 |886131.23 |
| MobileNet |0.75 |192 |Same as Inception |74.97% |93.50% |2.75 |20.80 |878126.80 |
| MobileNet |0.75 |160 |Same as Inception |74.02% |93.06% |2.92 |18.53 |872651.55 |
| MobileNet |0.50 |160 |Same as Inception |67.50% |89.37% |4.27 |11.91 |824833.36 |


## Mobilenet Accuracy Test Results - 80k dataset, Fine Tuning only fully connected layers. 

| Model | Width Multiplier | Image size | Preprocessing | Accuracy-Top1 |Accuracy-Top5 | Log loss | Image Time(ms) | Score |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|MobileNet |1.0 |224 |Same as Inception |70.09% |91.17% |3.63 |40.52 |825242.58 |


## Shallow MobileNet Accuracy Test Results - 80k dataset. To make MobileNet shallower, 3 layers of separable filters with feature size 14 × 14 × 512 are removed.

| Model | Width Multiplier | Image size | Preprocessing | Accuracy-Top1 |Accuracy-Top5 | Log loss | Image Time(ms) | Score |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Shallow MobileNet |1.0 |224 |Same as Inception |55.04% |80.40% |7.44 |31.75 |652383.53 |


### Prepare data

1. Decompress all the sysnets into a single training data directory. An example on how to do it can be found next; please make sure to update the location where all the sysnets.tar files are located.
```
python untar_sysnets.py
```
 
2. Create a new extended_ground_truth.csv file. Make sure to update the location where Imagenet Synsets directory is located.
```
python extended_ground_truth.py
```

3  . Move images under a directories tree, where each directory represents a class

```
MoveFilesIntoCategoriesAll.ipynb
```
```
root/
                  |->training/
                     |->1/
                     |     |-> 00****1.JPEG
                     |     L-> 00****n.JPEG
                     |->2/
                     |     |-> 00****1.JPEG
                     |     L-> 00****n.JPEG
                     .
                     .
                     .
                     L->200/
                           |-> 00****1.JPEG
                           L-> 00****n.JPEG
```

### Clean the dataset

4. Prepare images data. It's common to find errors where images cannot be read and/or rehsaped, resulting in an error when trying to transform them into tfrecords.
    The following Jupyter notebook can be used to find all the conflicting files, once found you can delete them from the dataset.
```
FindCorruptedFiles.ipynb
```

### Train the network

1. Convert data into tfrecord. Split the data in train and validation sets by passing the value of the validation_size, once your network is tuned, you can use all the data for training (e.g. --validation_size 0.01).
```
python preprocess_img_dir/create_tfrecord.py --validation_size 0.3
```

### Select what type of training you want, depending on the accuracy, and inference time needs

2.1 Train the network, Fine Tune all layers
```
python train.py --dataset_dir=/home/ubuntu/movidius/train --labels_file=/home/ubuntu/movidius/train/labels.txt --num_epochs 15 --image_size 224 --num_classes 200 --checkpoint_path=./models/checkpoints/mobilenet/01_224/mobilenet_v1_1.0_224.ckpt --checkpoint_exclude_scopes="MobilenetV1/Logits, MobilenetV1/AuxLogits" --log_dir=./tflog/full_run/01_224 --batch_size 16 --preprocessing inception --model_name mobilenet_v1 --tb_logdir=./TB_logdir/full_run/01_224
```

2.2 Train the network, Fine Tune only fully connected layers (Reduce training time by 50% or more at expense of accuracy)
```
python train.py --dataset_dir=/home/ubuntu/movidius/train --labels_file=/home/ubuntu/movidius/train/labels.txt --num_epochs 15 --image_size 224 --num_classes 200 --checkpoint_path=./models/checkpoints/mobilenet/01_224/mobilenet_v1_1.0_224.ckpt --checkpoint_exclude_scopes="MobilenetV1/Logits, MobilenetV1/AuxLogits" --log_dir=./tflog/full_run/01_224_FT --batch_size 16 --preprocessing inception --model_name mobilenet_v1 --tb_logdir=./TB_logdir/full_run/01_224_FT --trainable_scopes="MobilenetV1/Logits, MobilenetV1/AuxLogits"
```

2.3 Train the shallow network (As discussed in the Mobilenet paper, you get better results by using narrow models)
```
python train.py --dataset_dir=/home/ubuntu/movidius/train --labels_file=/home/ubuntu/movidius/train/labels.txt --num_epochs 15 --image_size 224 --num_classes 200 --log_dir=./tflog/full_run/custom/01_224 --batch_size 16 --preprocessing inception --model_name mobilenet_v1_custom --tb_logdir=./TB_logdir/full_run/custom/01_224
```

3. Visualize/Monitor your training accuracy and losses in TensorBoard
```
tensorboard --logdir ./TB_logdir/full_run/01_224
```

4. Evaluate the network
```
python eval.py --checkpoint_path ./tflog/full_run/01_224 --num_classes 200 --labels_file /home/ubuntu/movidius/train/labels.txt --dataset_dir /home/ubuntu/movidius/train --file_pattern movidius_%s_*.tfrecord --file_pattern_for_counting movidius --batch_size 16 --preprocessing_name inception --model_name mobilenet_v1 --image_size 224
```

5. Export the network
```
python export_inference_graph.py --model_name mobilenet_v1 --image_size 224 --batch_size 1 --num_classes 200 --ckpt_path ./tflog/full_run/01_224/model.ckpt-252435 --output_ckpt_path ./output/full_run/01_224/network
```

###Transfer your network.meta and weights files to your machine where NCS SDK is installed

6. Compile the network (e.g. compiled.graph)
```
mvNCCompile network.meta -w network -s 12 -in input -on output -o compiled.graph
```

7. Profile the network, (e.g. FLOPS, bandwidth, and processing time per layer)
```
mvNCProfile -in input -on output -s 12 -is 224 224 network.meta
```

8. Use the compiled.graph to make inferences. Make sure to update your path settings to point to the correct directories where your copiled.graph is stored. Also make sure to assign a name and location where the resulting file with the image inferences will be placed.
```
python inference.py path/to/datadir	
```

## TODO
- [ ] Train on different network architectures (e.g. DenseNet). Training and compiling was done, not unsuccessfully used for inference. 
- [ ] Support fot multi stick inference (e.g. 3 stick, inference time/3)
- [ ] Report results 

## Reference
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

[Technica-Corporation/TF-Movidius-Finetune](https://github.com/Technica-Corporation/TF-Movidius-Finetune)

[models/research/slim/nets/mobilenet_v1.md](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

[Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.](https://arxiv.org/abs/1409.0575)
