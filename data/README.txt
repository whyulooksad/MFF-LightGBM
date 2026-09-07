data 只有两条互不混用的通道：

developer：项目开发者的公开数据集、制模和评估数据。
  datasets/source/<数据集>/archives   官网压缩包原件。
  datasets/source/<数据集>/extracted  原样解压内容，不在这里清洗或改名。
  datasets/manifests                  来源、SHA-256、标签、group_id、split。
  datasets/prepared/{train,validation,test}  按分组建立的训练 PCAP 硬链接。
  workspace/features                  流特征、CLS 特征、融合特征。
  workspace/corpus                    RTD 与 LoRA 输入语料。
  workspace/evaluation                独立测试、消融、图表和资源测试。

runtime：用户检测时产生的数据。
  uploads                              用户刚上传的 PCAP/PCAPNG。
  tasks/<任务ID>                       本次特征、预测和汇总。
  state                                SQLite 等服务状态。

用户上传绝不能自动进入 developer 或参与训练。各叶子目录的 readme.txt 说明该目录允许放什么。
