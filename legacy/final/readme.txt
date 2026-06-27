LightGBM_Detector文件夹是孔腾珑给的文件，里面训练和测试时放在一个py文件下的

supcon_ae文件夹是李鑫的，这个我建议先别管，效果上有点问题，先把它去掉，特征直接接在检测器上跑，看看最后效果。

pcap -200.py是我用来从原始pcap中提取新pcap的，不用管他。


feature show.py是我的特征提取，输出保存到"\data\csv"。final_multiclass_features.csv是我的特征。

preprocess.py我已经处理好了，输出保存在"\data\json"。

extract_features.py要处理成：输入是preprocess.py的输出，输出是特征。

后面检测器你需要改改咯，你先把它跑通。之后你可以顺带问问孔腾珑它最后要的输出需要有什么东西。