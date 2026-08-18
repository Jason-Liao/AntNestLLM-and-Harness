# 外部合规语料（M3-语料 v4 增量）

来源与合规性：
- 中文经典：先秦典籍原文，著作权已进入公有领域（Public Domain）
- 英文技术文本：Python 软件基金会许可证（PSF License）允许再分发
- 工程常识：antNest 团队自写，无第三方版权

## 《论语》选（公有领域）

子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？
曾子曰：吾日三省吾身：为人谋而不忠乎？与朋友交而不信乎？传不习乎？
子曰：温故而知新，可以为师矣。
子曰：学而不思则罔，思而不学则殆。
子曰：知之为知之，不知为不知，是知也。
子曰：三人行，必有我师焉。择其善者而从之，其不善者而改之。

## 《道德经》选（公有领域）

道可道，非常道。名可名，非常名。无名天地之始；有名万物之母。
故常无欲，以观其妙；常有欲，以观其徼。此两者，同出而异名，同谓之玄。
玄之又玄，众妙之门。
上善若水。水善利万物而不争，处众人之所恶，故几于道。
合抱之木，生于毫末；九层之台，起于累土；千里之行，始于足下。

## 《孙子兵法》选（公有领域）

兵者，国之大事，死生之地，存亡之道，不可不察也。
故经之以五事，校之以计，而索其情：一曰道，二曰天，三曰地，四曰将，五曰法。
知彼知己者，百战不殆；不知彼而知己，一胜一负；不知彼，不知己，每战必殆。

## 工程常识（antNest 自写）

大模型的训练流程包括预训练、监督微调与强化学习三个阶段。
预训练从海量文本中学习语言规律；监督微调教会模型遵循指令；
强化学习以奖励信号对齐任务目标。
深度学习训练依赖梯度下降法：计算损失函数对参数的梯度，沿负梯度方向更新权重。
学习率调度通常采用预热加余弦退火策略，训练初期缓慢升温，后期逐步衰减。
过拟合的常见对策包括增加数据量、正则化、丢弃法与早停。
分布式训练将大模型切分到多张加速卡上，通过数据并行、张量并行与流水线并行提升吞吐。
软件工程质量保障的核心实践包括版本控制、自动化测试、持续集成与代码评审。

## Python PSF License (节选)

PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"),
and the Individual or Organization ("Licensee") accessing and otherwise
using this software ("Python") in source or binary form and its associated
documentation.
2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works,
distribute, and otherwise use Python alone or in any derivative version.
