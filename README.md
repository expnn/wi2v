# wi2v — WandB Images to Video

将 WandB 实验日志中的图像序列提取出来，合成为 MP4 视频，方便直观地观察训练过程中可视化输出的变化趋势。

支持断点续传、SHA256 完整性校验、多任务缓存管理。

## 安装
本项目依赖 ffmpeg，请确保 ffmpeg 已正确安装并添加到系统 PATH 环境变量中。

```bash
# 推荐：使用 uv
uv tool install https://github.com/expnn/wi2v.git

# 或者
git clone https://github.com/expnn/wi2v.git
cd wi2v
uv pip install -e .

# 或使用 pip
pip install -e .
```

## 快速开始

```bash
# 最简用法：无需指定 entity 和 project，自动检测
wi2v -r <RunID> -k <image_key>

# 完整参数
wi2v -e <entity> -p <project> -r <RunID> -k <image_key> -o output.mp4 --fps 30

# 重复运行自动断点续传
wi2v -r <RunID> -k <image_key>
```

## 命令参考

```
wi2v [convert]  [-e ENTITY] [-p PROJECT] -r RUN_ID [-k IMAGE_KEY] [-o OUTPUT] [--fps FPS]
wi2v list
wi2v clean       [-t TASK_ID | --all]
```

`convert` 是默认子命令，可省略。

### convert — 下载图像并合成视频

| 参数 | 简写 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--entity` | `-e` | 否 | 当前登录用户 | WandB 用户名或组织名 |
| `--project` | `-p` | 否 | 自动检测¹ | WandB 项目名 |
| `--run-id` | `-r` | **是** | — | WandB Run ID（URL 中的短哈希） |
| `--image-key` | `-k` | 否 | `val_images` | `wandb.log()` 中使用的 key |
| `--output` | `-o` | 否 | `experiment_timelapse.mp4` | 输出视频文件名 |
| `--fps` | | 否 | `10` | 视频帧率 |

> ¹ Project 三级解析策略：显式 `-p` → 上次缓存的 project → 遍历所有 project 搜索（~1.6s）

### list — 列出所有缓存任务

以表格形式展示每个任务的关键信息：

```
$ wi2v list
共 2 个任务:

  ID  Entity          Project     Run ID      Image Key                       已下载/总数          大小
----------------------------------------------------------------------------------------------------
   1  fusion-sim      msie        ubc036sg    vis/embedding/before/LEGACY_S...     55/56      51.9 MB
   2  fusion-sim      msie        abc123      val_images                          100/100       5.7 MB
```

### clean — 清空缓存

```bash
wi2v clean -t 1         # 清空 ID 为 1 的任务缓存
wi2v clean --all        # 清空所有任务缓存
```

## 断点续传

每次运行 `convert` 时，已成功下载的图像（SHA256 校验通过）会被自动跳过，仅下载缺失或损坏的帧。即使中途崩溃，下次运行同一命令即可继续。

每帧下载完成后立刻将 SHA256 写入清单文件（原子写入），确保数据完整性。

## 缓存结构

所有缓存存放在 `~/.cache/wi2v/` 下：

```
~/.cache/wi2v/
├── config.json                         # 本地配置（last_project）
├── index.json                          # 全任务索引
└── <entity>/
    └── <project>/
        └── <run_id>/
            └── <image_key>/
                ├── manifest.json        # 下载清单（含 SHA256）
                ├── step_000000.png
                ├── step_000010.png
                └── ...
```

每个任务（entity + project + run_id + image_key）拥有独立的缓存目录，互不干扰。

## 依赖

- Python ≥ 3.12
- [wandb](https://github.com/wandb/wandb) — WandB API 客户端
- [opencv-python](https://github.com/opencv/opencv-python) — 视频编码
- [tqdm](https://github.com/tqdm/tqdm) — 进度条

使用前请确保已登录 WandB：

```bash
wandb login
```