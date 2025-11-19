# NPU-TOP Pro 

NPU-TOP Pro 是一个基于终端的用户界面 (TUI) 工具，用于实时监控 NPU (Neural Processing Unit) 的状态。它提供了类似于 `nvtop` 或 `htop` 的体验，专门针对华为 Ascend NPU 设备设计。

## 功能特性

* **实时监控**：显示 NPU 的利用率、显存使用、温度、功耗和时钟频率。
* **矢量绘图**：使用 ASCII 字符绘制高精度的利用率和显存历史曲线。
* **进程管理**：列出当前占用 NPU 的进程详细信息（PID、用户、内存占用、启动命令）。
* **多卡支持**：自动检测并网格化显示多张 NPU 卡的状态。
* **高性能**：使用多线程并发获取 `npu-smi` 数据，减少界面卡顿。

## 依赖项

本工具依赖于 **Python 3.9+** 以及以下第三方库：

* `textual`: 用于构建 TUI 界面。
* `rich`: 用于终端富文本渲染。

此外，运行环境必须安装有华为的驱动工具包，即系统路径中需包含 `npu-smi` 命令。

## 安装与运行

### 1. 安装依赖

建议使用虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
pip install textual rich
```

### 2. 直接运行

在终端中直接执行脚本：

```bash
python npu-top.py
```

* 按 `q` 键退出程序。

## 打包指南 (构建独立可执行文件)

为了方便分发到没有安装 Python 环境的服务器上运行，可以使用 `PyInstaller` 将其打包为单文件可执行程序。

### 1. 安装 PyInstaller

```bash
pip install pyinstaller
```

### 2. 执行打包命令

运行以下命令生成单文件二进制包：

```bash
pyinstaller --onefile --name npu-top npu-top.py
```

* `--onefile`: 打包成单个文件。
* `--name npu-top`: 指定输出文件名为 `npu-top`。

### 3. 获取可执行文件

打包完成后，可执行文件位于 `dist/` 目录下：

```bash
ls -lh dist/npu-top
```

您可以将该文件复制到任何安装了 NPU 驱动的 Linux 服务器上直接运行：

```bash
./dist/npu-top
```

## 常见问题

**Q: 为什么显示数据为 0 或 N/A？**
A: 请确保当前用户有权限执行 `npu-smi` 命令。通常需要 root 权限或属于特定用户组。

**Q: 界面显示乱码？**
A: 请确保您的终端模拟器支持 UTF-8 编码，并且使用了支持 Box Drawing 字符的字体（如 Nerd Fonts, JetBrains Mono 等）。
