# 二类：物体分割 pipeline

**本 zip 的内容：** 代码（`segment.py`、`seg/`、`tools/`、`tests/`、`configs/`、运行脚本）、本 README，以及三段必做视频的结果 `results/<视频名>.json` 和 `results/<视频名>_overlay.mp4`。不包含客户的视频和 task sheet，也不包含模型权重（运行时自动下载，SAM 2 的下载命令见下）。

YOLO-World（开放词表检测；它看不见的类别自动改用 Grounding DINO）+ SAM 2（视频分割与跟踪）+ CLIP（整条轨迹的类别复核）+ 自己的 re-ID / 像素归属 / 防 id 交换规则 + 按 task sheet 数量和“机器人碰到或移动”筛选输出。没有对任何 mask、id 或物体清单做人工修正。调试时我把几张**输出** overlay 的截图给 AI 助手看过，用来描述错误现象（比如“多出一个 id”），它据此改的是通用规则，没有写任何 mask、id 或物体清单。代码由 AI 编码助手按我的要求编写，`configs/overrides.json` 也由它依据 task sheet 文字生成（没有看视频），我审阅代码约 2 小时。二类总共用时约 12 小时。

## 运行

```bash
pip install -r requirements.txt     # ultralytics (YOLO-World + CLIP), transformers (Grounding DINO), imageio-ffmpeg ...
SAM2_BUILD_CUDA=0 pip install "git+https://github.com/facebookresearch/sam2.git"   # CUDA 机器上去掉 SAM2_BUILD_CUDA=0
mkdir -p checkpoints && curl -L -o checkpoints/sam2.1_hiera_base_plus.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
# 每段视频一条命令（task_sheets.json 放在仓库根目录；设备自动选 cuda > mps > cpu）
python segment.py videos/12_fold_four_towels.mp4 --task-sheets task_sheets.json --out results
python segment.py videos/09_coil_and_stow_four_cables.mp4 --task-sheets task_sheets.json --out results
python segment.py videos/26_sort_cutlery_drawer_insert.mp4 --task-sheets task_sheets.json --out results
# Slurm（Brown Oscar）：每段一个 GPU 任务，跑完自动自查 + 报告
bash oscar_run.sh                          # 三段必做；OUT=results_full bash oscar_run.sh ... 换输出目录
bash run_all.sh videos auto base_plus      # 全部 30 段
# 自查
python tools/check_output.py results/12_fold_four_towels.json --video videos/12_fold_four_towels.mp4 \
    --overlay results/12_fold_four_towels_overlay.mp4 --dump results/check/12_fold_four_towels --sample 20
python tools/report.py results 12_fold_four_towels   # 期望 vs 实际数量、每个 id 的事件、CLIP 分数、检测决定
python tests/test_finalize.py        # 输出格式、数量分配、相关性规则（CPU）
python tests/test_tracker_mock.py    # 跟踪逻辑，假模型回放已知场景（CPU；没装 torch 时用 numpy 替身）
python tests/test_diagnostics.py     # 细长裁图两端、CLIP 阈值、通用数量解析、检测器分工
```

第一次运行会自动下载 YOLO-World、CLIP、（需要时）Grounding DINO 的权重。

输出：`results/<视频名>.json`、`results/<视频名>_overlay.mp4`；中间结果在 `results/work/<视频名>/`（逐帧 id 图、跟踪日志 `tracks.json`，可用 `--finalize-only` 只重跑输出步骤）。

改过 CLIP 裁图后，用 `--finalize-only --reverify` 复用跟踪结果、重新计算 CLIP 分数并生成输出；只用 `--finalize-only` 会沿用已有 `verify.json`。`--reverify` 会先清除旧复核文件，重算失败时会明确打印错误，输出阶段不再偷偷沿用旧分数。

检测诊断工具可用于任意 task sheet 对应的视频，不改检测词表或正式跟踪结果：

```bash
python tools/probe_det.py data/09_coil_and_stow_four_cables.mp4 --n 16 --device auto \
    --extra 'wire,cord,electrical cable,cable tie,velcro cable tie' --json results/probe_09.json
# 对照更大输入尺寸（仍然只是诊断）
python tools/probe_det.py data/09_coil_and_stow_four_cables.mp4 --n 16 --device auto --imgsz 1280 \
    --extra 'wire,cord,electrical cable,cable tie,velcro cable tie' --json results/probe_09_1280.json
```

工具读取类别竞争、NMS 和检测数量截断**之前**的每个 prompt 最高分，同时列出最终输出框的最高分；某个词没有输出框不等于它的原始分数为零。保存每帧的时间、分数、抽样成功数和运行设置，避免将抽样结果当成全视频结论。附加词仅用于此次诊断，不自动写入任何视频的配置。裁图补正方形、0.4 删除阈值（至少 3 张复核样本）、“the X”数量解析都作用于共用流程，不按视频编号或轨迹 id 加特例。

- 硬件：Brown CCV Oscar 的 GPU 节点（NVIDIA L40S / RTX A5000，每次分到哪张卡见 `logs/` 里 nvidia-smi 的输出），SAM 2.1 base_plus。也在 MacBook（Apple GPU，MPS）上跑通过，但慢很多：15 秒片段（750 帧）用 small 模型跟踪 1064 s，A5000 + base_plus 约 130 s。
- 每段用时（Oscar 一块 GPU，跟踪 + CLIP 复核 + 写输出，最终代码）：12_fold_four_towels（2484 帧）1340 s；09_coil_and_stow_four_cables（14957 帧，cable / strap 由 Grounding DINO 检测）6527 s；26_sort_cutlery_drawer_insert（8886 帧，木质餐具由 Grounding DINO 检测）7295 s。只重做输出（`--finalize-only`，CPU 节点）：12 号约 4 分钟，09 / 26 号约 25 分钟。

## 思路

**找哪些物体（`seg/vocab.py`）。** “任务相关”= 机器人碰到、移动、往里放或往上放的东西。task sheet 的 `object` 列正好是机器人操作的物体；“往里放 / 往上放”的容器从行文本里抽（`in / into / onto the …` 后面的名词，去掉 area、zone、table 这类地点词）。三段必做视频另有 `configs/overrides.json`，把 task sheet 的名字换成更好检测的 prompt，并合并同一物体的不同叫法：线缆卷好后 task sheet 叫 bundle，但它还是同一根线，所以 bundle → cable；餐具里 dinner spoon / teaspoon 检测器分不清，先都当 spoon，最后按 mask 面积分成两类：task sheet 写了各几把（“Place one dinner spoon” ×2、“Place one teaspoon” ×2）时，面积最大的那几把是 dinner spoon；没写时按面积自动找分界（26 号用自动分界得到 1 大 3 小，因为被遮挡的勺子面积偏小）。题目对任务物体的定义还包括“机器人碰到的东西”，而 task sheet 不会点名所有被碰到的东西（09 号放扎带的筐没有出现在任务文字里，第一次运行它被当成了 strap），所以每段视频都额外加几个通用容器词：basket、box、tray、bin、container（已有检测词里出现过的词不再加）；这些类别和 task sheet 物体一样，只有被机械臂碰到或移动过才会出现在输出里。机械臂也分割（类别 robot），因为它参与像素竞争，能减少物体 mask 漏到夹爪上。

**检测器分工（`seg/tracker.py` 的 `Detector.probe`，`seg/gdino.py`）。** 跟踪前先在均匀抽取的 16 帧上跑 YOLO-World；某个类别的所有 prompt 在这 16 帧上的最高分都不到 0.05，说明 YOLO-World 在这段视频里看不见它，这个类别整段视频改由 Grounding DINO（`IDEA-Research/grounding-dino-base`，transformers）检测，其他类别照旧。规则对所有视频一样，不按视频编号指定；分工结果写进 `tracks.json` 的 `detector` 字段。起因是 09 号：线缆 / 扎带在 YOLO-World 里原始分数最高不到 0.02（见下方诊断），提高阈值或输入尺寸都不够。

**分割和跟踪（`seg/tracker.py`）。** 视频切成 1 秒的窗口，每个窗口：

1. 上个窗口结束时还可见的物体，用它自己的 mask 作为 SAM 2 的提示继续跟。用 mask 而不是框，所以毛巾折叠、线缆卷起这类大变形时 id 不会断；
2. 窗口第一帧跑 YOLO-World（按类别分别去重，所以同一块区域既能出 robot 框也能出 towel 框），先处理机械臂，再处理物体：
   - 机械臂检测如果和已有的 robot mask 相连，就并进那只手臂（补全手臂，不开新 id）；
   - 物体检测框如果被**某一个**同类物体的 mask 盖住一半以上，算已有物体，只给类别投票；
   - 置信度低于 0.25 的检测，只有在最近 4 个窗口里同一位置（IoU ≥ 0.5）、同一类别已经出现过，才当作真实物体（被压在毛巾堆里的毛巾每个窗口都只有 0.1–0.2，偶发误检不会连续出现在同一处）；
   - 其余检测用 SAM 2 图像模式出 mask；mask 大部分落在同类已有 mask 里、或 30% 以上落在机械臂上的，拒绝（机械臂底座曾被检测成 towel）；
3. 新出现的物体先和“丢失”的物体做 re-ID：颜色直方图 + 和消失位置的距离 + 类别（有数量的类别之间不因类别不同扣分，检测器分不清叉 / 勺 / 刀；名额用满后，新检测一定是某个丢失的 id，取最像的那个，不再要求最低分）。每类 id 数有上限：`seg/vocab.py` 从 task sheet 文字里**自动**解析每类物体有几个（对 30 个任务都适用，不是手写数字）——标题里的数字（“Fold four towels”“six pairs of socks”→12）、写成 “one / a <物体>” 的行的 count（“Place one fork” ×2 → 2 把；同一物体的几行相加：两个 crate 各 6 个 grocery → 12）；只写 “the <物体>” 且这些行的 count 都是 1 的算 1 个（the insert、the wooden serving utensil）；“the day label” ×4、出现 “second / other <物体>” 的不给数（容易少算）。解析故意偏多：上限偏多只是少一层保护，偏少会丢掉真物体。有数量的类别**合用一个总名额**（26 号：叉 2 + 刀 2 + 勺 4 = 8），因为检测器经常把叉子叫成 spoon、把扎带叫成 cable，按单个类别卡名额会把真物体挡在外面（26 号第一次完整运行就是这样）。跟踪时的总名额是 task sheet 总数再加 25% 余量（一个误检的 id 不能把真物体挡在外面，精确数量在输出阶段再卡）。名额满了，新检测只能认回丢失的 id，认不上就丢；还没找齐时，低置信度检测不必等第二次出现就可以开 id。没有解析出数量的类别，上限是第一个窗口的数量 + 1，只在有同类丢失 id 时生效；
   - 检测范围：非机械臂的新候选，SAM mask 一半以上落在画面四边 10% 的边框里就不开 id（机械臂从边缘伸进来，不受限）。09 号把左下角一块黑色区域当成了 cable；
   - 一个有数量、还没找齐的类别的检测，连续 3 个窗口在同一位置被另一类物体“盖住”、而那个物体的 mask 比它大 1.6 倍以上：那个 mask 包住了两件挨在一起的东西（26 号一把刀和旁边的餐具），把它拆出来开新 id，并从原物体的提示 mask 里挖掉；往回补帧时它可以从原物体拿回像素。只出现一两次的局部检测不会触发拆分；
   - 机械臂的新检测并进已有手臂时，不覆盖已跟踪物体的像素（夹爪里拿着的线缆不会被并进手臂）；
   - 不是机械臂、但 SAM mask 超过画面 35% 的候选是桌面 / 背景，不开 id（09 号上一版把整块桌垫当成了 organizer，还吞掉了收纳盒的大部分格子）；输出阶段同样删掉面积中位数超过 35% 的轨迹；
   - 放在一个大得多的**其他类别**物体里 / 上的东西（托盘里的勺子、收纳格里的线缆卷）不算被它“盖住”：大物体的 mask 是小物体检测框的 4 倍以上、类别不同，就当两个物体处理；往回补帧时小物体也可以从这种“容器”那里拿回自己的像素。最早的版本没有这条，26 号视频放进托盘的餐具全部被当成托盘的一部分；
   - 类别投票：每个窗口、每个物体只投一票，投给盖在它上面置信度最高的那个检测类别（以前是所有检测的置信度相加，spoon 有 spoon / teaspoon 两个 prompt，票数天然多一倍）；
4. 一个物体从同类物体的 mask 里分出来时（从毛巾堆里拉出一条），把它的像素从那个旧物体的提示里减掉，避免两个 id 抢同一块布；
5. 新出现 / 重新出现的物体**反向传播**，一个窗口一个窗口往回补，直到第 0 帧或它连续一个窗口看不见为止；补帧时可以从同类物体（毛巾堆）里拿回自己的像素，但不占其他类别的像素；
6. 所有物体在窗口内正向传播，重叠像素按 SAM 2 logit 最大的归属（一个像素最多一个物体）。传播时逐帧检查“跳变”：某个物体的 mask 和自己上一帧几乎不重合、却大部分落在另一个同类物体上一帧的位置上，就判定 SAM 2 跳到了相似物体上，本窗口内关掉它，等重新检测到再 re-ID。每个物体形状变化最大的那一帧之后还会比一次颜色，对不上就撤掉它从那一帧起的像素（只对平时面积——整条轨迹的面积中位数——≥ 4000 像素的物体做：餐具这类细小物体的颜色直方图主要是噪声，26 号第一次运行里它们因此被反复撤掉 id 又认回。按当前帧面积判断试过，不行：12 号 15 s 时粉毛巾的 id 正漂到红毛巾上，那几帧 mask 偏小，检查被跳过，红毛巾此后一直没有自己的 id）。

每次检测的去向（覆盖、置信度不够、拒绝、新 id、re-ID）都写进 `work/<视频>/detections.jsonl`，`python tools/why.py <该文件> --cat towel` 可以按帧查看，用来查“为什么这个物体没被认出来”。

**CLIP 复核（`seg/verify.py`）。** 跟踪完后，每个物体在它面积较大的帧里取最多 8 张 mask 裁剪图（背景涂灰），用 CLIP（YOLO-World 内部用的同一个图文模型）和“a photo of a <prompt>”以及几个非物体文本（桌面、墙、阴影、机械臂……）比较。裁剪图先补成正方形（CLIP 预处理会从中间裁正方形，细长物体的两端——勺头、叉齿——会被切掉）。属于任务物体的概率平均 < 0.4 的轨迹删掉——除非它属于 task sheet 类别、并且被机械臂碰过而且移动过（深色桌垫上的黑色细线缆，CLIP 常常认成阴影或机械臂，09 号因此删掉过两根真线缆；机器人搬动过的东西按题目定义就是任务物体）。额外补的通用容器词不享受这条（容器又大又明显，CLIP 分数低多半是误检）。这条豁免在 26 号上造成了一个错误，见“已知失败情况”（真实物体目前最低 0.77，误检 0.23、0.34）；每个物体对每个类别的分数 = 0.2 × 检测投票 + 0.8 × CLIP 概率（各自归一化）。最初是各占一半；26 号上 YOLO-World 的投票把 8 件餐具的首选全部给了 spoon，而 CLIP（方形裁图之后）给出的是 4 勺 / 2 叉 / 1 刀 / 1 木质餐具，和 task sheet 的数量（勺 4、叉 2、刀 2、木质 1）除了一把没被跟踪到的刀之外完全一致——没有人工标注时，“哪个信号和 task sheet 数量自洽”是我们能用的检验，所以改为以 CLIP 为主。这样类别不只靠单帧检测，而是看整条轨迹。

**输出（`seg/finalize.py`）。** 先检查每条轨迹是否“连贯”：相邻帧 mask 的中位 IoU < 0.6 且一半以上的帧碎成 3 块以上的轨迹不是一个完整物体（实际遇到的是重复 id 被同类物体抢走像素后剩下的边角碎片），它的像素并入每帧里紧挨着它的同类物体，没有就去掉。然后在 task sheet 数量的约束下给每个物体定类别：按“物体质量（可见时长 × mask 连贯度 × CLIP 分数）× 类别分数”从高到低贪心分配，每个有数量的类别最多分到那么多个（检测器把 8 件餐具都叫 spoon 时，最像 spoon 的 4 件是 spoon，其余按分数去 fork / knife）。哪个类别都分不到的是重复 id：CLIP 分数 ≥ 0.5、且有大小相近的同类物体时，像素并入相邻同类物体（重复的毛巾碎片并回毛巾）；CLIP 分数低，或同类物体都比它大 4 倍以上（26 号托盘里一条被叫成 insert 的细长区域），直接去掉——并进去会让托盘的 mask 盖住放在里面的东西。没有数量的视频，每个物体直接取分数最高的类别。之后： 按题目对“任务相关”的定义（机器人碰到、移动、往里放或往上放的东西）再筛一次：task sheet 物体列里的类别（线缆、扎带、毛巾、餐具……），如果某条轨迹整段视频都没移动过（平滑后的中心最远不到 40 像素）、机械臂的 mask 也从没碰到过它，就是被误标成任务类别的东西（09 号角落的黑色区域），删掉；真实的任务物体一定会被机械臂碰到，所以不会被删。收纳盒、托盘这类容器不受这条限制。中心大部分时间都在画面边框里的轨迹同样删掉（检测范围）。删掉总可见时间 < 0.5 s 的 id（误检）；`note` 写首次出现时间、重新识别的时间和 CLIP 分数（试过按 mask 平均颜色自动写颜色名，金属餐具被叫成 green / cyan，不可靠，关掉了）；id 按首次出现顺序重排为 1..K（robot 排最后）；逐帧写 COCO RLE（没有 mask 的帧也列出）；overlay 由 ffmpeg 按原分辨率、原帧率逐帧编码，帧数与输入一致。

**为什么这样选。** SAM 2 的视频记忆是目前对变形物体保持一致 mask 最稳的现成方案，但它只跟踪被提示过的物体，而且长视频一次性加载会爆内存，所以用短窗口 + 每窗口重新提示；检测器只负责“发现”，不负责逐帧画框，这样检测器的抖动不影响 mask。re-ID 用颜色 + 位置而不是学习的特征，是因为同一视频里同类物体往往长得一样（两把叉子），真正能区分它们的是位置和“哪个还在场”。

**试过但没成功 / 改掉的做法。**（都在 15 秒毛巾片段上验证过）
- 每类 id 数硬上限（第一个窗口数量 + 1）：堆在一起的毛巾第一秒被少数，第 4 条毛巾出现时被直接丢掉。改成上限只用于强制复用丢失 id。
- “检测框被任何已有 mask 盖住 30% 就算已有物体”：机械臂横在毛巾上、相邻毛巾互相压着时，真实的新物体被吞掉。改成只看单个同类物体、且阈值 50%。
- 反向补帧只补上一个窗口：晚 3 秒才被认出的毛巾前 2 秒没有 mask。改成逐窗口一直往回补。
- 在**输出标签**上每 2 帧比较一次平均颜色（Lab）来抓 id 交换：确实能抓到粉 / 红毛巾互换（两者 Lab 距离约 70，同一条毛巾 1–12），但被拎起、翻折、进阴影的毛巾自身颜色也会变 37–63，导致大量误判、正确的毛巾反而缺帧，所以撤回，交换问题列入已知失败。

## 怎么检查自己

没有标准答案，所以查三类东西：

1. **格式**（`tools/check_output.py`）：每一帧按顺序出现、尺寸、RLE 能解码、id 都声明过、没有像素属于两个物体、overlay 的分辨率 / 帧率 / 帧数和输入一致。
2. **时序一致性**：50 fps 下相邻帧物体几乎不动，所以同一 id 相邻帧 mask 的 IoU 应接近 1。IoU < 0.5、面积突变 2.5 倍、碎成 > 3 块，基本都对应漏分割、漏到别的物体上、或 id 交换。每个 id 的消失 / 重现时刻也列出来，逐个看 overlay 判断 id 对不对。`--dump` 把这些时刻的 overlay 帧导出成图片。
3. **和 task sheet 数量对账**（`tools/report.py`）：从 task sheet 文字解析出的数量 vs 输出里的数量，见下方“已知失败情况”。26 号上，CLIP 给出的类别正好满足 task sheet 的数量、检测器投票不满足，这是把 CLIP 权重调到 0.8 的依据。
4. **人工抽查**（我本人看 overlay，不是 AI）：每次完整运行后看 `check_output.py` 导出的均匀抽样帧（每段 20 张）和被标记的帧，以及 overlay 视频里有问题的时间段，每次约 15 分钟。发现并据此改成通用规则的问题：12 号多出一个由毛巾碎片组成的 id；09 号把整块桌垫标成 organizer、把角落一块黑色区域标成 cable、把放扎带的筐标成 strap；26 号托盘里一条细长区域被标成第二个 insert、少一把刀。

此外有两组不需要 GPU 的测试：`test_finalize.py` 检查输出格式；`test_tracker_mock.py` 用假模型回放一个已知场景，覆盖这次排查中遇到的每种情况：叉子被遮挡后重现、一模一样的第二把叉子一直在场、刀晚 1.5 秒才被检测到、超出 id 上限的新毛巾、被叉子压住 1/3 的毛巾、置信度一直只有 0.15 的毛巾、被检测成 towel 的机械臂底座、SAM 2 跳到相邻毛巾上。断言每个物体全程只有一个 id、补帧从真正出现的那一帧开始。它们只验证逻辑，不代表模型效果。

## 已知失败情况

（以提交的输出为准；数量来自 `tools/report.py`，时间来自 `check_output.py` 的标记帧。）

**提交的输出出自哪一版。** 三段都由本 zip 里的代码完整跑出（跟踪 + CLIP 复核 + 输出），之后又用同一份代码 `--finalize-only` 重新生成了输出；`results/` 里的 JSON 和 overlay 就是这次的结果。开发过程中 26 号有一次完整重跑 1 小时 45 分钟只跑了 300 帧：每段视频新加的通用容器词（basket、bin、container）在 26 号里 YOLO-World 一个都看不见，被自动交给了 Grounding DINO，它在托盘和桌面上到处报“容器”；已改为只有 task sheet 里的类别才会交给 Grounding DINO。其余 27 段选做视频没有运行。

- **09_coil_and_stow_four_cables：少一根线缆，线缆 / 扎带可能互换。** task sheet 是 4 根线缆、4 条扎带，输出 4 条 strap、3 根 cable（另有 organizer、2 只机械臂）。YOLO-World 对线缆 / 扎带的原始分数在抽样帧上最高不到 0.02（cable 0.0014、strap 0.0040，输入尺寸 1280 也一样），改由 Grounding DINO 检测：卷起来的线缆能看见（“coiled cable” 16 帧里 11 帧 ≥ 0.30），单根线缆和扎带分数都很低（扎带最高 0.12）。CLIP 也分不清深色桌垫上的线缆和扎带——3 根 cable 的 CLIP 首选都是 strap，是扎带名额（4）用满后按数量分过去的；其中两根（id 5、7）CLIP 物体分数只有 0.24 / 0.36，因为机械臂碰过并移动过它们才保留下来。所以同一类里“哪根是线缆、哪根是扎带”可能有互换。id 1 strap 全程几乎不动，可能是放扎带的筐【没有人工核对】。id 3 strap 的 mask 不稳（中位帧间 IoU 0.91）。早期版本：第一次运行分到的节点没有 ffmpeg；只用 YOLO-World 时一根线缆都没有找到；机械臂合并没有“不覆盖它拿着的物体”时，左臂 mask 大量破洞（中位帧间 IoU 0.79），标记帧 5832，现在 2502。
- **26_sort_cutlery_drawer_insert：多了一个错误的 “container”。** task sheet 物体的数量都对：叉 2、刀 2、勺 4（dinner 2 + tea 2）、木质餐具 1、托盘 1，另有 2 只机械臂。早期版本少一把刀（旁边一件餐具的 mask 把它也包了进去，检测一直被当成那件餐具的一部分），加了“拆分贴在一起的物体”之后找回（id 10，21 s 起）；第二把叉子 86 s 才第一次出现（id 11）。标记帧 3820 个，集中在 142–177 s：餐具一件件放进托盘、托盘被推走时，勺子的 id 不稳（id 3 dinner spoon 155.6–167.0 s 有 467 个标记帧，id 9 teaspoon 断开 63 次，id 1 木质餐具 155–161 s）。跟踪时多开的 2 个勺子 id 按数量并回了相邻的勺子；**多了一个 id 6 “container”**：它其实是右臂下半截的一部分。木质餐具这一类 YOLO-World 看不见、交给了 Grounding DINO，Grounding DINO 把这段机械臂检测成了“木质勺子 / 铲子”（检测投票 wooden utensil 9.0）；CLIP 认为它不是任务物体（0.30），但它属于 task sheet 类别、又一直被机械臂“带着移动”，于是按上面的豁免留了下来；木质餐具的名额已被真的那件占用，最后被分到 container。它覆盖的那块机械臂在输出里归它，不归 robot；162 s 后消失。
  - 应对的想法（没来得及用在提交的输出上）：机械臂自己的一部分会一直挨着机械臂，真实物体总有一段时间单独躺在桌上或托盘里——所以“被机械臂移动过”的豁免只给至少 20% 的时间不挨着机械臂的轨迹；一直挨着的按 CLIP 分数正常删除。另外，跟踪时“候选 mask 有 30% 以上落在机械臂上就拒绝”这条没能挡住它，可以再加一条：候选在接下来几个窗口里一直和机械臂一起运动、从不分开，就并回机械臂。餐具的类别以 CLIP 为主，没有人工逐个核对。
- **12_fold_four_towels：** 4 条毛巾 + 2 只机械臂，数量对；每条毛巾覆盖 2228–2484 帧，中位帧间 IoU ≥ 0.99，标记帧 162 个。11.7–14.0 s 两条叠在一起的毛巾 id 互换（id 3 在这段有 37 个标记帧；SAM 2 把一条毛巾的 id 逐渐“爬”到另一条上，逐帧跳变检查抓不到）。16–23 s 一条毛巾每秒都被颜色检查撤掉又认回（被拎起、翻折时颜色变化大），之后这条毛巾被开了一个新 id 接替；旧 id 只剩碎片，作为“不连贯轨迹”并进了相邻毛巾——最终数量正确，但这条毛巾前半段的 mask 来自新 id 往回补帧。
- 结构性问题：
  - 两个同类同色物体同时被遮挡后在交换的位置重现，re-ID 会给反（只靠位置）。
  - 反向补帧中途跟丢（物体在补帧区间里被完全挡住），更早的帧仍然没有 mask。
  - 堆在一起的同类物体在被单独拿起之前和堆一起作为一个物体；补帧时按 SAM 2 反向跟踪结果从堆里分出来，分界不一定准。
  - SAM 2 的 mask 可能逐渐漏到相邻的相似物体上（叠放的毛巾、并排的线缆）。
  - dinner spoon / teaspoon 靠面积区分，勺子一部分被挡住时面积会偏小。

## 再给一周和一块 GPU

1. 用检测框定期校正 SAM 2 的 mask 漂移（框和 mask 不一致时重新提示），并对每个 id 做时间上的 mask 平滑。
2. re-ID 换成 DINOv2 特征 + 运动连续性（被夹爪带走的物体，跟着夹爪轨迹找），并在消失期间记录“被哪个物体盖住 / 放进了哪个容器”，重现时优先在那里匹配。
3. 试 SAM 3 的文本提示视频分割替代 YOLO-World + SAM 2 的组合，比较漏检和 id 切换数量。
4. 人工精标每段 20 帧左右，量 mask IoU 和 id 切换次数，用来调窗口长度、阈值，而不是只靠无监督的一致性指标。
5. 把逐帧检测改成只在“出现新东西”的帧附近跑，降低每段视频的耗时。
6. 区分“机械臂的一部分”和“被机械臂拿着的物体”：按整条轨迹挨着机械臂的时间比例（见 26 号的已知失败）、以及机械臂 mask 的完整性（两只手臂各自应是一个连通的整体），而不是只看单个窗口。
7. 09 号这类检测器看不见的细长物体（扎带、细线）：用 SAM 自动分割出所有区域再让 CLIP 判断类别，或者用少量标注帧微调一个检测器；以及把机械臂 mask 的破洞问题（跟踪时机械臂吞掉夹爪里的物体）单独处理。
