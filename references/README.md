# 获取参考 notebook

训练脚本读取本目录的两份原 notebook，再对特定单元进行适配。原作者代码在其来源页面获取；本仓库不重新分发 notebook 或比赛数据。

1. 下载 [NB0036.ipynb](https://github.com/Rajneesh-Tiwari/FlightRank-2025-Aeroclub-RecSys-Cup/blob/3ddbf8a46dd35f2d625fd29f0e46bb1b97d1add6/src/NB0036.ipynb) 至 `references/NB0036.ipynb`。SHA-256：`2cd7a46fb270aef8d363bc8fa0be3eb8866463b0e34b1e6a15a356b0e113041b`。
2. 从 [公开 baseline](https://www.kaggle.com/code/ka1242/xgboost-ranker-with-polars) 下载 version 5（scriptVersionId 250405784）至 `references/baseline.ipynb`。本次文件 SHA-256：`b174b9062d31da98706b7a4afe80d6153ae84520f6df92ed73623778c04d5c66`。Kaggle 不同下载方式可能导出不同格式；需使用与该哈希相同的 notebook 文件。

可用 `shasum -a 256 references/*.ipynb` 检查。请遵守原来源和比赛的数据使用条款。
