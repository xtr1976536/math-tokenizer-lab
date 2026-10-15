# GPU 租赁平台比较（2026-10-16 查询）

本实验优先选择单卡 RTX 4090／4090D 24GB。作者本来就在 4090D 上运行。
已固定的 PyTorch 2.3.1 + CUDA 12.1 适合 Ada／Ampere；不要直接把较新
Blackwell GPU 的硬件优势当作此环境可直接使用的保证。

| 地区 | 平台 | 官网公开报价 | 选型与隐私依据 |
|---|---|---|---|
| 国内 | [AutoDL](https://www.autodl.com/home?channel=00000000000000000000000040000001&version=7.60.5.106) | RTX 4090 24GB：¥1.88／小时；3090 24GB：¥1.32／小时 | 国内优先方案。其[隐私政策](https://www.autodl.com/docs/privacy_policy/)说明未经许可不访问业务数据；使用自己的实例和 SSH，不公开数据服务端口。 |
| 国内 | [Featurize](https://www.featurize.cn/) | RTX 4090：低至 ¥1.87／小时，页面标注 60GB 主存、750GB 磁盘 | 中文界面、主存与磁盘配置适合本任务。结合[服务协议](https://docs.featurize.cn/docs/agreement/agreement)查看实例、账号与公开端口的使用条款。 |
| 国外 | [Runpod Secure Cloud](https://www.runpod.io/pricing) | RTX 4090 24GB：$0.89／小时；RTX A6000 48GB：$0.59／小时 | 国外优先方案，选 Secure Cloud。其[安全文档](https://docs.runpod.io/references/security-and-compliance)说明容器隔离、宿主访问限制及数据中心合作方要求。 |
| 国外 | [Vast.ai](https://vast.ai/pricing) | 按实时机器报价，没有统一固定 4090 价格 | 选 On-Demand、Secure Cloud／Trusted Datacenters，核对可靠率、CPU、内存及存储费用。[安全分级](https://vast.ai/compliance)区分普通 Verified Hosts 与可信数据中心。 |

报价为官网页面展示的小时价或起价，机器可用性、会员、存储与税费以选择
实例时显示的总价为准。上表比较公开政策与产品分级，不将容器隔离解释为
宿主机在技术上绝对无法读取运行中的数据。

我的选择顺序：国内先 AutoDL 4090 与 Featurize 4090 比较实际可用实例；
若选国外，先 Runpod Secure Cloud 4090。Vast.ai 更适合愿意逐台核对配置与
主机可靠性的选择。A6000 虽便宜且显存大，单纯训练速度通常不应按显存容量
判断，先跑本包的 GPU smoke 和一小段计时再确定长任务预算。

本包没有平台密钥、自动购卡、自动上传、远程日志上报或自动支付代码。
