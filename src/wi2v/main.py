import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time

import ffmpeg
import numpy as np
import wandb
import wandb.errors
from PIL import Image
from tqdm import tqdm

CACHE_ROOT = os.path.expanduser("~/.cache/wi2v")


# ---- 工具函数 ----


def _validate_image(path: str) -> bool:
    """验证图像文件是否完整可读。"""
    try:
        with Image.open(path) as img:
            img.convert('RGB')
        return True
    except Exception:
        return False


def _sha256_file(path: str) -> str:
    """计算文件的 SHA256 哈希。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: str) -> dict:
    """加载 JSON 文件，不存在则返回空 dict。"""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(path: str, data: dict) -> None:
    """原子写入 JSON（先写 .tmp 再 rename）。"""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _config_path() -> str:
    return os.path.join(CACHE_ROOT, "config.json")


def _load_config() -> dict:
    return _load_json(_config_path())


def _save_config(config: dict) -> None:
    os.makedirs(CACHE_ROOT, exist_ok=True)
    _save_json(_config_path(), config)


def _task_dir(entity: str, project: str, run_id: str, image_key: str) -> str:
    """由任务参数生成相对缓存目录路径。"""
    safe_key = image_key.replace("/", "-")
    return os.path.join(entity, project, run_id, safe_key)


def _task_slug(entity: str, project: str, run_id: str, image_key: str) -> str:
    """任务的可读标识字符串。"""
    safe_key = image_key.replace("/", "-")
    return f"{entity}/{project}/{run_id}/{safe_key}"


def _index_path() -> str:
    return os.path.join(CACHE_ROOT, "index.json")


def _load_index() -> dict:
    return _load_json(_index_path())


def _save_index(index: dict) -> None:
    os.makedirs(CACHE_ROOT, exist_ok=True)
    _save_json(_index_path(), index)


def _register_task(
    entity: str, project: str, run_id: str, image_key: str
) -> tuple[int, str]:
    """在全局索引中登记任务。返回 (task_id, 缓存目录)。"""
    index = _load_index()
    slug = _task_slug(entity, project, run_id, image_key)
    tasks = index.setdefault("tasks", {})

    if slug not in tasks:
        next_id = int(index.get("next_id", 1))
        tasks[slug] = {
            "id": next_id,
            "entity": entity,
            "project": project,
            "run_id": run_id,
            "image_key": image_key,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        index["next_id"] = next_id + 1
    t = tasks[slug]
    t["last_accessed"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_index(index)

    dir_path = os.path.join(CACHE_ROOT, _task_dir(entity, project, run_id, image_key))
    os.makedirs(dir_path, exist_ok=True)
    return t["id"], dir_path


def _find_task_by_id(task_id: int) -> tuple[str, dict] | None:
    """根据 task_id 查找任务，返回 (slug, info) 或 None。"""
    index = _load_index()
    for slug, info in index.get("tasks", {}).items():
        if info.get("id") == task_id:
            return slug, info
    return None


def _dir_size_mb(work_dir: str) -> float:
    """计算目录下所有文件的总大小（MB）。"""
    total = 0
    if os.path.isdir(work_dir):
        for f in os.listdir(work_dir):
            fp = os.path.join(work_dir, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def _format_size(size_mb: float) -> str:
    if size_mb < 1:
        return f"{size_mb * 1024:.0f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb / 1024:.1f} GB"


# ---- convert 子命令 ----


def _resolve_project(
    api, entity: str, run_id: str, explicit_project: str | None = None
) -> str | None:
    """根据 run_id 解析 project 名。

    1. 显式指定 -> 直接使用
    2. 本地缓存 last_project -> 先用缓存尝试
    3. 缓存中找不到 -> 扫描所有 project
    """
    # 显式指定，直接返回
    if explicit_project is not None:
        return explicit_project

    # 尝试本地缓存
    config = _load_config()
    cached = config.get("last_project")
    if cached:
        try:
            api.run(f"{entity}/{cached}/{run_id}")
            print(f"使用上次的 Project: {cached}")
            return cached
        except wandb.errors.CommError:
            pass  # run 不在缓存的项目中，继续扫描

    # 扫描所有 project
    print(f"正在搜索 Run {run_id} 所属的 Project...")
    for project in api.projects(entity=entity):
        try:
            for run in api.runs(f"{entity}/{project.name}", filters={"name": run_id}):
                if run.id.endswith(run_id) or run.name == run_id:
                    print(f"找到 Project: {project.name}")
                    return project.name
        except wandb.errors.CommError:
            pass

    return None


def _do_convert(
    entity: str,
    project: str,
    run_id: str,
    image_key: str,
    output: str,
    fps: int,
) -> None:
    """核心转换逻辑：下载图像并合成视频。所有参数须已解析完毕。"""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    print(f"正在从 Run {run_id} 中提取图像信息...")
    history = run.scan_history(keys=[image_key, "_step"])
    rows = [row for row in history if image_key in row]
    rows.sort(key=lambda x: x["_step"])

    if not rows:
        print("未找到指定的图像 Key，请检查配置。")
        return

    # 注册任务，获得独立的缓存目录
    task_id, work_dir = _register_task(entity, project, run_id, image_key)
    manifest_path = os.path.join(work_dir, "manifest.json")
    manifest = _load_json(manifest_path)
    cached = manifest.get("images", {})

    total_frames = len(rows)

    # 检查哪些 step 需要下载
    to_download: list[tuple[str, dict]] = []
    for row in rows:
        step = str(row["_step"])
        entry = cached.get(step)
        if entry:
            filepath = os.path.join(work_dir, entry["file"])
            if os.path.exists(filepath) and _sha256_file(filepath) == entry["sha256"]:
                if _validate_image(filepath):
                    continue
                # 缓存的图像损坏，删除并重新下载
                os.remove(filepath)
                del manifest["images"][step]
        to_download.append((step, row))

    print(
        f"[任务 #{task_id}] 共 {total_frames} 帧，已缓存 {total_frames - len(to_download)} 帧，待下载 {len(to_download)} 帧"
    )

    # 下载缺失 / 损坏的图像
    for step, row in tqdm(to_download, desc="下载", unit="帧"):
        img_data = row[image_key]
        remote_path = img_data["filenames"][0]

        ext = os.path.splitext(remote_path)[1] or ".png"
        local_name = f"step_{step.zfill(6)}{ext}"
        local_path = os.path.join(work_dir, local_name)

        try:
            fh = run.file(remote_path).download(replace=True, root=work_dir)
            downloaded_path = fh.name
            if os.path.abspath(downloaded_path) != os.path.abspath(local_path):
                if os.path.exists(local_path):
                    os.remove(local_path)
                os.rename(downloaded_path, local_path)

            if not _validate_image(local_path):
                tqdm.write(f"下载文件损坏 step {step}，已删除")
                os.remove(local_path)
                continue

            sha = _sha256_file(local_path)
            manifest.setdefault("images", {})[step] = {
                "file": local_name,
                "sha256": sha,
            }
            manifest["total_frames"] = total_frames
            _save_json(manifest_path, manifest)
        except wandb.errors.CommError as e:
            tqdm.write(f"下载失败 step {step}: {e}")
            for p in (local_path, local_path + ".part"):
                if os.path.exists(p):
                    os.remove(p)

    # 最终更新 total_frames
    manifest = _load_json(manifest_path)
    manifest["total_frames"] = total_frames
    _save_json(manifest_path, manifest)

    # 收集最终有效的图像路径
    manifest = _load_json(manifest_path)
    cached = manifest.get("images", {})
    image_paths: list[str] = []
    missing: list[str] = []

    for row in rows:
        step = str(row["_step"])
        entry = cached.get(step)
        if entry:
            filepath = os.path.join(work_dir, entry["file"])
            if os.path.exists(filepath) and _sha256_file(filepath) == entry["sha256"]:
                image_paths.append(filepath)
                continue
        missing.append(step)

    if missing:
        print(f"警告: {len(missing)} 帧下载失败，已跳过")

    if not image_paths:
        print("没有成功下载任何图像，无法合成视频。")
        return

    # 合成视频
    print(f"正在合成视频: {output}（共 {len(image_paths)} 帧）...")

    if not shutil.which("ffmpeg"):
        print("错误: 未找到 ffmpeg，请先安装: brew install ffmpeg")
        return

    try:
        first_img = np.array(Image.open(image_paths[0]).convert('RGB'))
    except Exception as e:
        print(f"错误: 无法读取首帧 {image_paths[0]}: {e}")
        return
    h, w = first_img.shape[:2]

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{w}x{h}', framerate=fps)
        .output(output, vcodec='libx264', pix_fmt='yuv420p', crf=23, movflags='+faststart')
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )

    def _read_stderr():
        for line in process.stderr:
            tqdm.write(line.decode().rstrip())

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    try:
        for path in image_paths:
            try:
                img = np.array(Image.open(path).convert('RGB'))
                if img.shape[:2] != (h, w):
                    print(f"警告: {path} 尺寸异常 {img.shape[:2]}，期望 ({h},{w})，使用黑色帧代替")
                    img = np.zeros((h, w, 3), dtype=np.uint8)
                    os.remove(path)
            except Exception as e:
                print(f"警告: 无法读取 {path}，使用黑色帧代替。\n{e}")
                img = np.zeros((h, w, 3), dtype=np.uint8)
                try:
                    os.remove(path)
                except OSError:
                    pass
            process.stdin.write(img.tobytes())
    finally:
        process.stdin.close()
        process.wait()
        stderr_thread.join(timeout=5)

    print("完成！")


def cmd_convert(args):
    entity = args.entity
    project = args.project
    run_id = args.run_id
    image_key = args.image_key

    api = wandb.Api()
    if entity is None:
        entity = api.default_entity
        print(f"自动获取 Entity: {entity}")

    project = _resolve_project(api, entity, run_id, explicit_project=project)
    if project is None:
        print(f"错误: 未找到 Run {run_id} 所属的 Project，请用 -p 显式指定。")
        return

    config = _load_config()
    config["last_project"] = project
    _save_config(config)

    _do_convert(entity, project, run_id, image_key, args.output, args.fps)


# ---- list 子命令 ----


def cmd_list(args):
    index = _load_index().get("tasks", {})
    if not index:
        print("（无缓存任务）")
        return

    print(f"共 {len(index)} 个任务:\n")

    # 表头
    header = f"{'ID':>4}  {'Entity':<14}  {'Project':<10}  {'Run ID':<10}  {'Image Key':<30}  {'已下载/总数':>10}  {'大小':>10}"
    print(header)
    print("-" * len(header))

    for slug, info in sorted(index.items(), key=lambda kv: kv[1].get("id", 0)):
        tid = info.get("id", "?")
        entity = info["entity"]
        project = info["project"]
        rid = info["run_id"]
        key = info["image_key"]
        work_dir = os.path.join(CACHE_ROOT, _task_dir(entity, project, rid, key))

        manifest = _load_json(os.path.join(work_dir, "manifest.json"))
        downloaded = len(manifest.get("images", {}))
        total = manifest.get("total_frames", "?")
        progress = f"{downloaded}/{total}"

        size_mb = _dir_size_mb(work_dir)
        size_str = _format_size(size_mb)

        # 截断过长的 key
        key_display = key if len(key) <= 30 else key[:27] + "..."

        print(
            f"{tid:>4}  {entity:<14}  {project:<10}  {rid:<10}  {key_display:<30}  {progress:>10}  {size_str:>10}"
        )

    print()


# ---- clean 子命令 ----


def cmd_clean(args):
    if args.all:
        index = _load_index().get("tasks", {})
        if not index:
            print("没有可清理的缓存。")
            return
        for slug, info in index.items():
            work_dir = os.path.join(
                CACHE_ROOT,
                _task_dir(
                    info["entity"], info["project"], info["run_id"], info["image_key"]
                ),
            )
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
                print(f"  已清理: #{info.get('id', '?')} {slug}")
        _save_index(_index_path(), {})
        idx_path = _index_path()
        if os.path.exists(idx_path):
            os.remove(idx_path)
        print("所有缓存已清空。")
        return

    # 通过 task_id 清理
    task_id = args.task_id
    if task_id is None:
        print("请指定 --task-id <ID> 来清理特定任务，或使用 --all 清空所有。")
        return

    result = _find_task_by_id(task_id)
    if result is None:
        print(f"未找到 ID 为 {task_id} 的任务。使用 list 命令查看所有任务。")
        return

    slug, info = result
    work_dir = os.path.join(
        CACHE_ROOT,
        _task_dir(info["entity"], info["project"], info["run_id"], info["image_key"]),
    )
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        print(f"已清理: #{task_id} {slug}")
    else:
        print(f"缓存目录不存在: #{task_id} {slug}")

    # 从索引中移除
    index = _load_index()
    if slug in index.get("tasks", {}):
        del index["tasks"][slug]
        _save_index(index)


# ---- redo 子命令 ----


def cmd_redo(args):
    """根据 task_id 重新执行转换。"""
    task_id = args.task_id
    result = _find_task_by_id(task_id)
    if result is None:
        print(f"未找到 ID 为 {task_id} 的任务。使用 list 命令查看所有任务。")
        return

    slug, info = result
    entity = info["entity"]
    project = info["project"]
    run_id = info["run_id"]
    image_key = info["image_key"]

    print(f"重新执行任务 #{task_id}: {slug}")

    # 更新 last_project
    config = _load_config()
    config["last_project"] = project
    _save_config(config)

    _do_convert(entity, project, run_id, image_key, args.output, args.fps)


# ---- 主入口 ----


def main():
    parser = argparse.ArgumentParser(description="从 WandB 日志中提取图像并合成为视频")
    sub = parser.add_subparsers(dest="command", help="子命令")

    # convert
    p_conv = sub.add_parser("convert", aliases=["c"], help="执行图像下载与视频合成")
    p_conv.add_argument(
        "--entity", "-e", default=None, help="WandB 用户名（默认：自动获取）"
    )
    p_conv.add_argument(
        "--project", "-p", default=None, help="WandB 项目名（默认：上次使用的项目）"
    )
    p_conv.add_argument("--run-id", "-r", required=True, help="WandB Run ID")
    p_conv.add_argument(
        "--image-key",
        "-k",
        default="val_images",
        help="wandb.log() 中使用的图像 key（默认：val_images）",
    )
    p_conv.add_argument(
        "--output", "-o", default="experiment_timelapse.mp4", help="输出视频文件名"
    )
    p_conv.add_argument("--fps", type=int, default=10, help="视频帧率（默认：10）")
    p_conv.set_defaults(func=cmd_convert)

    # list
    p_list = sub.add_parser("list", aliases=["ls"], help="列出所有缓存任务")
    p_list.set_defaults(func=cmd_list)

    # clean
    p_clean = sub.add_parser("clean", aliases=["rm"], help="清空缓存")
    p_clean.add_argument(
        "--task-id", "-t", type=int, default=None, help="要清理的任务 ID"
    )
    p_clean.add_argument("--all", "-a", action="store_true", help="清空所有任务的缓存")
    p_clean.set_defaults(func=cmd_clean)

    # redo
    p_redo = sub.add_parser("redo", help="重新执行指定任务的转换")
    p_redo.add_argument(
        "--task-id", "-t", type=int, required=True, help="任务 ID"
    )
    p_redo.add_argument(
        "--output", "-o", default="experiment_timelapse.mp4", help="输出视频文件名"
    )
    p_redo.add_argument("--fps", type=int, default=10, help="视频帧率（默认：10）")
    p_redo.set_defaults(func=cmd_redo)

    # 无子命令时默认执行 convert（向后兼容）
    KNOWN_COMMANDS = {"convert", "c", "list", "ls", "clean", "rm", "redo", "-h", "--help"}
    if len(sys.argv) == 1 or (len(sys.argv) >= 2 and sys.argv[1] not in KNOWN_COMMANDS):
        sys.argv.insert(1, "convert")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
