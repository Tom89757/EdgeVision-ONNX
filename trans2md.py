#!/usr/bin/env python3
"""trans2md.py — 把源码/文本文件转成 Markdown 文档。

用法：
    python trans2md.py test.java             # 单文件：同目录生成 test.md
    python trans2md.py folder java py sh     # 目录：同级 <folder名>2md 下转换 java/py/sh
    python trans2md.py folder                # 目录：转换所有常见代码/文本格式

目录模式：
    在 folder 同级目录创建 <folder名>2md/（如 android_app -> android_app2md），
    保持原目录结构与文件名（文件名追加 .md 后缀），内容用带语言标签的 ``` 代码块包裹。

单文件模式：
    去掉原扩展名，在与源文件同层目录生成 <主名>.md（如 test.java -> test.md）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 目录模式下未指定扩展名时的默认常见代码/文本格式
DEFAULT_EXTS = {
    "java", "py", "xml", "json", "c", "cpp", "cc", "h", "hpp",
    "js", "ts", "jsx", "tsx", "css", "html", "htm",
    "yml", "yaml", "properties", "gradle", "kts", "bat", "sh", "cmake",
    "md", "txt", "csv", "sql", "ini", "toml", "conf", "cfg",
    "kt", "go", "rs", "swift", "rb", "php", "lua",
}

# 扫描时跳过的目录名（构建产物、依赖等）
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "build", ".gradle",
    "node_modules", ".idea", ".vscode", "dist",
    "legacy_datasets", ".codex_logs",
}

# 扩展名 -> Markdown 代码块语言标签
LANG_MAP = {
    "py": "python",
    "sh": "bash",
    "bat": "bat",
    "yml": "yaml",
    "yaml": "yaml",
    "md": "markdown",
    "txt": "text",
    "cc": "cpp",
    "hpp": "cpp",
    "js": "javascript",
    "ts": "typescript",
    "jsx": "jsx",
    "tsx": "tsx",
    "htm": "html",
    "gradle": "groovy",
    "kts": "kotlin",
    "kt": "kotlin",
    "rs": "rust",
    "rb": "ruby",
    "properties": "properties",
    "cmake": "cmake",
    "toml": "toml",
    "ini": "ini",
    "conf": "ini",
    "cfg": "ini",
    "csv": "csv",
    "sql": "sql",
}


def code_lang(ext: str) -> str:
    """返回代码块语言标签；未知扩展名回退为扩展名本身。"""
    return LANG_MAP.get(ext, ext)


def read_text_safe(path: Path) -> str | None:
    """按 UTF-8 优先、GBK 回退读取文本；二进制/异常返回 None。"""
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except (OSError, ValueError):
            return None
    return None


def build_md(source_name: str, ext: str, content: str) -> str:
    """构造 Markdown 内容：标题 + 带语言标签的代码块。"""
    content = content.replace("\r\n", "\n").rstrip("\n")
    return f"# {source_name}\n\n```{code_lang(ext)}\n{content}\n```\n"


def convert_file(file: Path) -> int:
    """单文件模式：同目录生成 <主名>.md。"""
    ext = file.suffix.lower().lstrip(".")
    content = read_text_safe(file)
    if content is None:
        print(f"[错误] 无法读取（可能是二进制或编码不支持）：{file}", file=sys.stderr)
        return 1
    out = file.with_suffix(".md")
    out.write_text(build_md(file.name, ext, content), encoding="utf-8")
    print(f"[OK] {file.name} -> {out}")
    return 0


def convert_dir(folder: Path, exts_args: list[str]) -> int:
    """目录模式：同级 <folder名>2md/ 下按目录结构生成 md。"""
    exts = {e.lower().lstrip(".") for e in exts_args} if exts_args else DEFAULT_EXTS
    out_dir = folder.parent / (folder.name + "2md")
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    for source in sorted(folder.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(folder)
        if any(part in EXCLUDED_DIRS or part.endswith("2md") for part in rel.parts):
            continue
        ext = source.suffix.lower().lstrip(".")
        if ext not in exts:
            continue

        content = read_text_safe(source)
        if content is None:
            print(f"[跳过] 无法读取（可能是二进制或编码不支持）：{rel}")
            skipped += 1
            continue

        md_rel = Path(str(rel) + ".md")
        md_path = out_dir / md_rel
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(build_md(source.name, ext, content), encoding="utf-8")
        converted += 1
        print(f"[OK] {rel} -> {out_dir.name}/{md_rel}")

    print(f"\n[完成] 转换 {converted} 个文件，跳过 {skipped} 个，输出目录：{out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="把源码/文本文件转成 Markdown 文档")
    parser.add_argument("path", nargs="?", help="要转换的文件或目录（必填）")
    parser.add_argument("exts", nargs="*", help="目录模式下要转换的扩展名；省略时使用默认常见格式")
    args = parser.parse_args()

    if args.path is None:
        parser.print_help()
        print("\n[错误] 请指定一个文件或目录，例如：", file=sys.stderr)
        print("  python trans2md.py test.java", file=sys.stderr)
        print("  python trans2md.py android_app java py sh", file=sys.stderr)
        return 1

    target = Path(args.path).expanduser().resolve()
    if target.is_file():
        return convert_file(target)
    if target.is_dir():
        return convert_dir(target, args.exts)

    print(f"[错误] 路径不存在：{target}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
