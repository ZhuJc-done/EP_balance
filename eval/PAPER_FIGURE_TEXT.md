# Paper-ready figure text

## English caption

**Figure X: Expert-routing imbalance in Qwen3-30B-A3B on math and code
workloads.** We evaluate a frozen Qwen3-30B-A3B checkpoint (48 MoE layers,
128 logical experts, and top-8 routing) on equal-token-budget DAPO-Math and
StarCoderData corpora. Panels (a) and (b) show the logical-expert
token-assignment shares for captured micro-batch occurrence 0 on DAPO-Math and
StarCoderData, respectively; rows correspond to MoE layers, columns to expert
IDs, and darker cells to larger assignment shares. Panel (c) reports the
maximum expert load divided by the mean expert load at each layer. Solid lines
are means over 93 captured occurrences and shaded
regions denote the P10--P90 range. A value of 1 indicates perfectly uniform
routing. The concentrated hotspots and consistently large max/mean ratios
demonstrate strong, workload-dependent routing imbalance.

## English body text

Figure X shows that learned MoE routing is far from uniform even when no
synthetic router skew is introduced. Across layers, the hottest expert receives
on average 7.341 times the mean expert load on DAPO-Math and 7.078 times the
mean on StarCoderData. The mean per-occurrence ratio peaks at 12.934 in layer 8
for DAPO-Math and at 8.845 in layer 3 for StarCoderData. Meanwhile, many cells
in the snapshot heatmaps remain lightly colored or empty, indicating experts
that receive few or no assignments in that occurrence.

This behavior is consistent with content-dependent expert specialization:
patterns that occur frequently in the current workload can repeatedly obtain
high router scores for a small subset of experts, creating hot experts while
leaving others underutilized. The different layer-wise profiles for math and
code further indicate that the imbalance depends on workload content rather
than only on static expert placement. Under fixed expert parallel placement,
such logical-expert skew can translate into stragglers and wasted capacity.
The routing counts establish the imbalance itself; identifying the exact
semantic capability of each expert would additionally require token-level
attribution analysis.

## 中文图释

**图 X：Qwen3-30B-A3B 在数学与代码负载上的专家路由不均衡。**
实验使用冻结的 Qwen3-30B-A3B 检查点（48 个 MoE 层、128 个逻辑专家、
每个 token 路由至 8 个专家），并采用 token 预算相同的 DAPO-Math 与
StarCoderData 语料。子图 (a) 和 (b) 分别展示 DAPO-Math 与 StarCoderData
在第 0 个 micro-batch occurrence 的逻辑专家 token 分配占比；纵轴为 MoE
层，横轴为专家编号，颜色越深表示该专家接收的 token 占比越高。子图 (c)
展示每层最大专家负载与专家平均负载之比；实线为 93 次
采样的均值，阴影为 P10--P90 区间。比值为 1 表示完全均匀的路由。集中出现的
热点及显著大于 1 的 max/mean 比值说明路由具有明显且与负载相关的不均衡。

## 中文正文

图 X 表明，即使未人为注入 router skew，MoE 学习得到的路由结果通常也并不
均匀。DAPO-Math 中最热专家相对平均专家的负载在各层平均达到 7.341 倍，
StarCoderData 中达到 7.078 倍；逐 occurrence 均值的峰值分别出现在第 8 层
（12.934 倍）和第 3 层（8.845 倍）。与此同时，快照热力图中存在大量浅色
甚至空白单元，说明部分专家在该批次中仅接收少量或没有接收 token。

这一现象与内容相关的专家专门化机制相一致：当前负载中频繁出现的模式可能
持续对少数专家产生较高的路由分数，从而形成热点专家，而其他专家则利用率
较低。数学与代码数据呈现不同的逐层曲线，也说明不均衡不仅由静态专家放置
决定，还会随输入内容发生变化。在固定的专家并行放置下，这种逻辑专家负载
偏斜可能进一步转化为设备等待和容量浪费。需要注意的是，当前路由计数能够
证明负载不均衡，但若要确定每个专家具体擅长的语义模式，还需要进一步进行
token 级归因分析。
