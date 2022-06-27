
<a name="Nsacv"></a>
### batchwise的验证数据生成
这里使用了imagenet2012中val数据集的数据，分成不同batch并经过onnx推理后将输入输出存入npy文件中，与初赛一样，便于最后的推理验证
```
cd data/
python batch_data_gen.py 
```
