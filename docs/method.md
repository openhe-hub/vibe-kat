# KAT Keypoint Token & Action Token 完整流程

## 整体架构

KAT 把 in-context imitation learning 拆成三层：

- **Perception 层**：把 RGB-D 观测变成 K 个 3D keypoint token
- **Action representation 层**：把 end-effector 的 6-DoF pose 变成 3 个 3D 点
- **LLM 层**：把 N 个 demo 的 (keypoints, actions) 配对塞进 prompt，让 LLM 续写新场景下的 actions

整个方法的优雅来自一个根本设计：**perception 的输出和 action 的表示都是同一种数据类型——3D 点坐标**。这让 LLM 的 input 和 output 完全同构，它的工作简化成 "given some 3D points, predict other 3D points" 的纯几何 pattern completion。

---

## Keypoint Token 流程

### Step 1: 输入 RGB-D 观测

机器人用 wrist camera（装在 gripper 上的 RealSense D435）拍一张 RGB-D 图像，resize 到约 224×224，有 RGB 三通道和 depth 一通道。

### Step 2: 喂进 DINO-ViT 得到 dense feature map

KAT 用 DINO-ViT-S/8：patch size 8，feature 维度 384。224×224 的图被切成 28×28 = **784 个 patch**，每个 patch 出来一个 384 维向量。

输出是一个 shape 为 `(28, 28, 384)` 的 feature map，可以摊平成 784 个 patch features 的列表。

### Step 3: 用 Best Buddies Nearest Neighbours (BBNN) 跨 demo 选 salient descriptors

KAT 不是在单张图上选 keypoint，而是**跨多张 demo 图**联合选择——找出在所有 demo 里都一致出现的视觉特征。

具体做法：对每一对 demo 图 (A, B)，在 784 个 patch features 之间找 **mutual nearest neighbours**（互为最近邻的 patch 对）。如果 patch a 是 B 中与 A 的 patch a 最相似的，同时 a 也是 A 中与 B 的那个 patch 最相似的——这对 patch 就是 best buddies。

跨所有 demo pair 收集到的 best buddy descriptors 数量是变长的（可能几百到几千个）。然后用 **K-means** 把这些 best buddy descriptors 聚类到恰好 K 个 cluster center——这些就是 K 个 salient descriptors。

**关键**：BBNN 是核心选择机制（找跨 demo 一致的特征），K-means 只是后处理（把变长的 BBNN 结果压缩到固定 K 个）。BBNN 的隐含假设是"在多张不同场景图中都能互相匹配的 patch，大概率是任务相关的稳定视觉特征"。

### Step 4: 用 salient descriptors 在新图中定位 2D keypoints

拿到 K 个 salient descriptors 后，对任何一张新图（demo 或 query），在它的 784 个 patch features 中，找与每个 salient descriptor 最相似（cosine similarity 最高）的 patch，记下它在 28×28 grid 上的位置 `(row, col)`。转回原图像素坐标（patch 中心）：

```
pixel_x = col * 8 + 4
pixel_y = row * 8 + 4
```

得到 K 个 2D 像素位置。

### Step 5: 用 depth 把 2D 提升到 3D（在相机坐标系下）

对每个 2D 像素位置 `(x, y)`：

1. 在 depth map 里查这个位置的深度 `d`
2. 用相机 intrinsics 反投影到相机坐标系：

```
P_camera = d · K⁻¹ · [x, y, 1]ᵀ
```

得到 K 个 3D 点，**全部在 wrist camera frame 下**——也就是说原点是相机光心，z 轴指向相机看的方向。

**关键点**：KAT **不**把这些点转到 world frame 或 robot base frame。它停在相机坐标系，因为它从不使用相机的 extrinsic calibration。这是 KAT 一个核心的工程妥协，下面会展开为什么。

### Step 6: 序列化成文本 token

把 K 个 3D 点格式化成纯文本，每个数字保留 3 位小数：

```
keypoint_0: [0.342, -0.128, 0.715]
keypoint_1: [0.401, -0.095, 0.682]
...
keypoint_K: [0.388, 0.156, 0.703]
```

这就是塞进 LLM prompt 的 keypoint tokens。

### 关于 keypoint 的几个关键事实

**1. LLM 看到的只有坐标，没有 DINO feature**

DINO 的 384 维 feature 在 K-means 完成之后就被丢掉了。LLM 完全不知道 keypoint_0 在语义上是什么——它只看到一串数字。LLM 做的是纯粹的 spatial pattern completion，不是语义推理。

**2. Keypoint 索引天然跨 demo 一致**

因为所有图共享同一组 K 个 salient descriptors（从 BBNN 聚类得来），每张图里 keypoint_k 都是"与第 k 个 salient descriptor 最匹配的 patch"。这自动保证了跨 demo 的语义一致性——keypoint_0 在所有图里都对应同一个视觉特征。

但这个一致性依赖 BBNN 能找到真正任务相关的 best buddies。如果 demo 间场景差异太大（如 RLBench 中物体位置完全随机），BBNN 倾向于选择**静态不变的特征**（桌面纹理、机械臂关节）而非任务物体。这是 KAT 在 unstaged 场景下失败的根源。

**3. K 是固定的**

不管场景里有几个物体，输出永远是 K 个 keypoint。这是为了让 LLM prompt 的格式固定。BBNN 本身会输出变长的 best buddy 集合，所以需要 K-means 后处理把数量压缩到恰好 K 个。

---

## Action Token 流程

### Step 1: 取 end-effector 的 6-DoF pose

在 demo 录制或 inference 时，end-effector 在某个时刻有一个 6-DoF pose，包含 position（3 DoF）+ rotation（3 DoF）。

### Step 2: 把 pose 转成 3 个 3D 点

KAT 不直接用 quaternion 或 Euler angle 表示 rotation，而是用一个独特的 **3-point representation**：

1. **Origin 点**：end-effector 的中心位置 `p_origin`
2. **Forward 点**：在 end-effector 的 x 轴方向上伸出一个固定长度 L（比如 5cm）的点，`p_forward = p_origin + L · x_axis`
3. **Up 点**：在 y 轴方向上伸出长度 L 的点，`p_up = p_origin + L · y_axis`

这三个 3D 点**唯一确定**了 6-DoF pose。从这三个点可以用 Gram-Schmidt 恢复完整的 rotation matrix：

```
x_axis = (p_forward - p_origin) / L
y_axis = (p_up - p_origin) / L  (再正交化)
z_axis = x_axis × y_axis
```

这三个点也在 **wrist camera frame** 下，和 keypoint 同一个坐标系。

### Step 3: 加上 gripper state

3 个 3D 点 = 9 个数字，再加 1 个 gripper 开合状态（0 或 1），一个 action token 一共 10 个数字。

### Step 4: 序列化成文本

```
action_0: [x_o, y_o, z_o, x_f, y_f, z_f, x_u, y_u, z_u, g]
action_1: [...]
...
```

一条 demo 的 trajectory 通常被 subsample 成 10-20 个 keyframe action token，而不是 dense trajectory。dense trajectory 会让 prompt 爆炸且 LLM 抓不到 structure。

### 为什么是 3 个点而不是 quaternion 或 Euler

这个设计有三层理由，从浅到深：

**表面层：和 keypoint input 保持 representation 同构**

LLM 看到的整个 prompt 里，所有数字都是"3D 点的坐标"——keypoint 是 3D 点，action 也是 3 个 3D 点。LLM 不需要学会"前 3 维是 position、后 4 维是 quaternion"这种异构格式，它只是在做 point-to-point 的 pattern completion。

**中间层：等价于 6D rotation representation**

3 个 3D 点（origin + 两个 basis 端点）在数学上等价于 Zhou et al. 2019 提出的 6D rotation representation——后者被广泛认为是 deep learning 里最适合 rotation 的表示方法（连续、无 singularity、无约束）。KAT 继承了 6D rotation 的所有优点：

- 没有 quaternion 的 unit norm 约束
- 没有 Euler angle 的 gimbal lock
- 没有 quaternion 的 double cover（q 和 -q 表示同一旋转的歧义）
- 数值范围连续，对插值友好

**深层：让 rigid body motion 表现为"所有点的一致变换"**

考虑 end-effector 沿 z 轴下降 10cm。用 position + quaternion 表示，前 3 维变了，后 4 维不变——LLM 必须学会这种异构变化。用 3-point 表示，三个点的 z 坐标各自减少 0.1——LLM 看到的是 9 个数字里都有同一个空间偏移，这正好是 LLM 擅长的"local pattern + interpolation"。

物理上这是对的：rigid body 平移时构成它的所有点都平移同样的量。**3-point 表示让物理直觉直接映射到 LLM 的数字模式，KAT 不需要 LLM 理解任何 "rotation 怎么作用在 position 上" 的 SO(3) 群作用**。

### 为什么是 3 个点而不是 2 个或 4 个

- **2 个点不够**：origin + 一个方向只能确定 5 DoF（位置 3 + 方向 2），绕那个方向轴的旋转无法确定
- **3 个点刚好**：origin + 两个方向，Gram-Schmidt 之后可以确定完整 SO(3)
- **4 个点多余**：多一个点不增加任何信息，反而增加 LLM 出错概率

3 是最小完备数。

---

## 坐标系问题（最容易被忽略但最关键）

KAT 的所有数字——keypoint 和 action——都活在 **wrist camera 坐标系**下。这不是个细节，是 KAT 整个方法成立的基础。

### 为什么必须是 camera frame

如果用 world frame 或 robot base frame，需要相机的 extrinsic calibration（相机相对 base 的位姿）。KAT 故意避开这个依赖——它从不需要 calibration，部署门槛因此大大降低。

更深的理由：在 base frame 下，**同一个相对 task** 在不同 demo 里会表现出完全不同的数字模式。比如"抓取杯子"这个 task，杯子在桌面 A 位置时 LLM 看到一组数字，在 B 位置时看到另一组数字——LLM 没有能力做 SE(3) 变换把这两组数字标准化。

但在 camera frame 下（特别是 wrist camera frame），相机本身就装在 gripper 上，所以 end-effector 在相机 frame 下的位置是**永远固定**的。物体相对 end-effector 的相对几何关系直接表现为相机看到的物体位置。**同一个相对 task 在不同 demo 里会产生相似的数字模式**，LLM 才有 pattern 可以续写。

KAT 的 camera frame 选择本质上是把"相对几何关系"内嵌到坐标系里——LLM 不需要做任何 frame normalization，这件事被坐标系本身免费做了。

### 这个选择的代价

camera frame 是非惯性的（跟着 end-effector 一起动），代价至少有三个：

1. **任务必须在视野内完成**：超出 wrist camera 视野的目标无法表达
2. **不能融合多视角**：因为没用 extrinsic calibration，第二个相机的数据没法和第一个对齐
3. **不能直接做 closed-loop replanning**：每帧相机自己在动，坐标系每帧都变，上一帧的数字模式下一帧不直接适用——这是 KAT 严格 open-loop 的根本原因，不是设计偏好

---

## 完整流程组装：LLM 看到的是什么

把上面所有步骤组合起来，LLM 在 inference 时看到的 prompt 大致是：

```
[System: 描述任务格式和坐标系约定]

Demo 1:
  Scene:
    keypoint_0: [x, y, z]
    keypoint_1: [x, y, z]
    ...
    keypoint_K: [x, y, z]
  Actions:
    action_0: [x_o, y_o, z_o, x_f, y_f, z_f, x_u, y_u, z_u, g]
    action_1: [...]
    ...

Demo 2:
  ...

Demo N:
  ...

Query:
  Scene:
    keypoint_0: [x, y, z]
    ...
    keypoint_K: [x, y, z]
  Actions:
    [LLM 续写到这里]
```

LLM 续写的 actions 数字被 parse 出来，每 9 个数字恢复成 3 个 3D 点，用 Gram-Schmidt 转回 6-DoF pose，加上 gripper state，机器人按这个 waypoint 序列开环执行。

---

## 把所有约束串起来看 KAT 的设计逻辑

KAT 的所有看似独立的选择，其实是一根逻辑链上的环节：

1. **目标**：用 zero-shot 的 text LLM 做 ICIL → 必须把所有信息压缩成短文本
2. **压缩 input**：丢掉 DINO feature，只保留坐标 → keypoint 必须是固定 K 个点
3. **固定 K 个点**：用 BBNN 选跨 demo 一致的 salient features，再 K-means 压缩到 K 个
4. **BBNN 依赖跨 demo 一致性**：demo 间差异大时选到静态背景特征 → perception 脆弱性的根源
5. **避开 calibration**：所有坐标停在 camera frame
6. **camera frame 非惯性**：必须严格 open-loop，任务必须在视野内
7. **action 和 keypoint 共享 representation**：用 3-point 表示 6-DoF pose，让 LLM 做纯几何插值
8. **LLM 不能做 SE(3) reasoning**：camera frame 的"相对几何内嵌"是 pattern-learnable 的关键

每一个选择单独看都合理，但合在一起把 KAT 的适用范围锁定在了 "桌面 + 单视角 + open-loop + 物体在视野内 + perception 干净的场景"。这就是为什么 KAT 在 staged lab 环境 work、在 unstaged sim work 不了——不是 bug，是这一整套约束的必然结果。
