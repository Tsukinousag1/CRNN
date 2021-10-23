# CRNN
paper：[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)


--------------------------------------
### 1. create your own dataset

Process different text texts, and store image path and label label in LMDB database in one-to-one correspondence. Here, the text format I processed is as follows：
```
./9/7/186_densely_20626.jpg 20626
./9/7/185_acidic_759.jpg 759
...
```
Extract useful information from text 186_densely_20626.jpg , build target path and correspond to label: densely

![](https://user-images.githubusercontent.com/60562159/138550828-c71102e2-c7b5-4041-b091-17b0ff4b6382.png)


----------------------------------------
### 2. dataset

Remove the following characters when reading LMDB data
```
b'******' ——→ '******'
```

------------------------------------------
### 3.train 

![](https://user-images.githubusercontent.com/60562159/138551016-3a9d3e34-17c4-475c-a950-3635c575f323.png)




