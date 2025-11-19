import subprocess
import sys
import math
import os
import pwd
import concurrent.futures
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, DataTable, Label, Static
from textual import work
from rich.text import Text
from rich.style import Style

# ==========================================
# Configuration
# ==========================================
REFRESH_RATE = 1.0
HISTORY_LEN = 60    # Horizontal axis length

# ==========================================
# Data Structures
# ==========================================

@dataclass
class NPUData:
    id: str
    name: str = "Unknown"
    health: str = "OK"
    bus_id: str = "N/A"
    power: float = 0.0
    temp: int = 0
    
    # Usage Stats
    ai_core_util: int = 0
    ai_cpu_util: int = 0    
    ctrl_cpu_util: int = 0  
    mem_bw_util: int = 0    
    
    # Static/Capacity Info
    ai_core_count: int = 0
    clock_freq: int = 0     # MHz
    
    # Memory
    memory_used: int = 0
    memory_total: int = 0
    hbm_used: int = 0
    hbm_total: int = 0
    
    history_util: deque = field(default_factory=lambda: deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN))
    history_mem: deque = field(default_factory=lambda: deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN))
        
@dataclass
class ProcessData:
    npu_id: str
    pid: str
    name: str
    memory: float
    cmdline: str = ""
    user: str = "Unknown"

# ==========================================
# Core: ASCII Vector Plotting Engine
# ==========================================

class AsciiCanvas:
    """
    A character-grid canvas for drawing sharp polylines with dynamic axis.
    Uses connectivity logic to handle complex overlaps (vertical merges, crossings).
    """
    
    # 1. 定义字符到方向的映射 (Top, Bottom, Left, Right)
    # T=1, B=2, L=4, R=8 (位掩码思想，但在Python中用Set更直观)
    CHAR_MAP = {
        " ": set(),
        "─": {"L", "R"},
        "│": {"T", "B"},
        "┌": {"B", "R"},
        "┐": {"B", "L"},
        "└": {"T", "R"},
        "┘": {"T", "L"},
        "├": {"T", "B", "R"},
        "┤": {"T", "B", "L"},
        "┬": {"L", "R", "B"},
        "┴": {"L", "R", "T"},
        "┼": {"T", "B", "L", "R"},
        # 特殊处理：如果遇到无法识别的字符，默认无连接
    }

    # 2. 反向映射：将方向集合转回字符 (FrozenSet作为Key)
    # 预计算反向映射以提高性能
    CONN_TO_CHAR = {frozenset(v): k for k, v in CHAR_MAP.items()}

    def __init__(self, width: int, height: int):
        self.width = max(1, width)
        self.height = max(1, height)
        self.padding_left = 4
        self.grid = [[(None, None) for _ in range(self.width)] for _ in range(self.height)]

    def clear(self):
        self.grid = [[(None, None) for _ in range(self.width)] for _ in range(self.height)]

    def _scale_y(self, val: float, min_v: float, max_v: float) -> int:
        if max_v == min_v: return 0
        val = max(min_v, min(val, max_v))
        ratio = (val - min_v) / (max_v - min_v)
        row_from_bottom = int(ratio * (self.height - 1))
        return (self.height - 1) - row_from_bottom

    def draw_y_axis(self, min_v=0, max_v=100):
        # Clear padding
        for r in range(self.height):
            for c in range(self.padding_left):
                self.grid[r][c] = (" ", None)

        labels = []
        if self.height >= 8:
            labels = [100, 75, 50, 25, 0]
        elif self.height >= 5:
            labels = [100, 50, 0]
        else:
            labels = [100, 0]

        used_rows = set()
        for val in labels:
            row = self._scale_y(val, min_v, max_v)
            if row in used_rows: continue
            label_str = f"{val:>3}"
            if 0 <= row < self.height:
                for i, char in enumerate(label_str):
                    if i < self.padding_left:
                        self.grid[row][i] = (char, "dim grey50")
                used_rows.add(row)

    def plot_line(self, data: List[float], color: str, min_v=0, max_v=100):
        if not data: return
        available_width = self.width - self.padding_left
        if available_width <= 0: return

        start_idx = max(0, len(data) - available_width)
        visible_data = list(data)[start_idx:]
        x_offset = self.width - len(visible_data)
        style = Style(color=color, bold=True)

        for i in range(len(visible_data) - 1):
            col = x_offset + i
            if col >= self.width - 1: break
            if col < self.padding_left: continue

            y0 = visible_data[i]
            y1 = visible_data[i+1]
            r0 = self._scale_y(y0, min_v, max_v)
            r1 = self._scale_y(y1, min_v, max_v)

            # 绘制垂直线段和转角
            if r0 == r1:
                self._draw_char(r0, col, "─", style)
            elif r0 > r1: # 上升 (row index 减小)
                self._draw_char(r0, col, "┘", style) # Start point (Bottom-Left connection?? No, Up-Left)
                                                     # 修正逻辑：r0是起点，向右走到r1。
                                                     # 实际上 ASCII 绘图通常是 point-to-point。
                                                     # 这里采用：起点是 "上+左"(┘)，终点是 "下+右"(┌)，中间是 "竖"(│)
                for r in range(r1 + 1, r0):
                    self._draw_char(r, col, "│", style)
                self._draw_char(r1, col, "┌", style)
            elif r0 < r1: # 下降
                self._draw_char(r0, col, "┐", style)
                for r in range(r0 + 1, r1):
                    self._draw_char(r, col, "│", style)
                self._draw_char(r1, col, "└", style)

        if visible_data:
            last_y = visible_data[-1]
            last_r = self._scale_y(last_y, min_v, max_v)
            last_col = self.width - 1
            self._draw_char(last_r, last_col, "─", style)

    def _draw_char(self, row, col, new_char, new_style):
        if not (0 <= row < self.height and 0 <= col < self.width):
            return
        if col < self.padding_left:
            return

        existing_char, existing_style = self.grid[row][col]

        final_char = new_char
        final_style = new_style

        # 如果该位置已经有字符，进行连通性合并
        if existing_char and existing_char != " ":
            # 1. 获取现有连接方向
            conns_existing = self.CHAR_MAP.get(existing_char, set())
            # 2. 获取新字符连接方向
            conns_new = self.CHAR_MAP.get(new_char, set())
            
            # 3. 合并方向
            merged_conns = conns_existing | conns_new
            
            # 4. 查找合并后的字符
            # dict.get default return "┼" if complex overlap not found, or keep existing logic
            matched_char = self.CONN_TO_CHAR.get(frozenset(merged_conns))
            
            if matched_char:
                final_char = matched_char
            else:
                # 如果组合出来的方向很怪（比如只有 Top），默认回退到新字符或由现有字符决定
                # 但对于标准 Box Drawing，上面的 Map 几乎覆盖了所有组合
                final_char = "┼" 

            # 5. 颜色混合逻辑 (Dithering)
            if existing_style != new_style:
                # 如果完全重叠（形状没变），或者是垂直/水平重叠，为了区分不同线条，使用网格交错色
                if col % 2 == 0:
                    final_style = existing_style
                else:
                    final_style = new_style
                
                # 优化：如果合并后的形状明显是新线条主导的（例如原有 ─，新来 │，变成了 ┼），
                # 这种复杂的判断比较难，保持简单的交错色通常视觉效果最好。

        self.grid[row][col] = (final_char, final_style)

    def render(self) -> Text:
        result = Text()
        for r in range(self.height):
            for c in range(self.width):
                char, style = self.grid[r][c]
                if char:
                    result.append(char, style=style)
                else:
                    # Grid background styling
                    if r == 0 or r == self.height - 1:
                        result.append("┄", style="dim grey15")
                    elif r % 4 == 0:
                        result.append("┄", style="dim grey11")
                    else:
                        result.append(" ", style=None)
            result.append("\n")
        return result
# ==========================================
# UI Components
# ==========================================

class PolylineWidget(Static):
    def __init__(self, title: str):
        super().__init__()
        self.chart_title = title
        # Removed fixed canvas_height, will use dynamic height
        self.canvas = None

    def update_plot(self, util_data, mem_data):
        # Use content_size to get the actual size allocated by layout
        if not self.content_size: return
        
        w = self.content_size.width
        h = self.content_size.height
        
        # Prevent crash on zero height
        if h <= 0 or w <= 0: return

        # Recreate canvas if size changed
        if not self.canvas or self.canvas.width != w or self.canvas.height != h:
            self.canvas = AsciiCanvas(w, h)
            
        self.canvas.clear()
        
        # 1. Draw Y Axis (0-100)
        self.canvas.draw_y_axis(0, 100)
        
        # 2. Draw Curves
        self.canvas.plot_line(mem_data, "cyan", 0, 100)
        self.canvas.plot_line(util_data, "green", 0, 100)
        
        self.update(self.canvas.render())

class NPUCard(Container):
    def __init__(self, npu_id: str):
        super().__init__(classes="npu-card")
        self.npu_id = npu_id
        self.header_label = Label("", classes="card-header")
        self.sub_header_label = Label("", classes="card-sub-header")
        self.plot_widget = PolylineWidget(f"NPU {npu_id}")
        self.footer_label = Label("", classes="card-footer")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield self.header_label
            yield self.sub_header_label
            yield Container(self.plot_widget, classes="plot-container")
            yield self.footer_label

    def update(self, data: NPUData):
        # 1. Metrics Calculation
        mem_gb = data.hbm_used / 1024
        mem_total = data.hbm_total / 1024
        mem_pct = int((data.hbm_used / data.hbm_total * 100)) if data.hbm_total else 0

        # 2. Header: ID, Name, Freq, Health
        health_color = "green" if data.health == "OK" else "red"
        freq_str = f"{data.clock_freq}MHz" if data.clock_freq > 0 else "---"
        
        self.header_label.update(
            f"[{health_color}]●[/] NPU [bold white]{data.id}[/] {data.name} [dim]@{freq_str}[/]"
        )
        
        # 3. Sub-Header: Temp, Power, AI Util, HBM Util
        temp_style = "bold red" if data.temp > 70 else "bold green"
        self.sub_header_label.update(
            f"[{temp_style}]{data.temp}°C[/]  {data.power}W  [bold green]AI:{data.ai_core_util}%[/]  [bold cyan]HBM:{mem_pct}%[/]"
        )

        # 4. Footer: Detailed Stats & NVTOP-style Bar
        footer_text = Text()
        
        # Row 1: Bandwidth & CPUs
        footer_text.append("BW: ", style="dim")
        footer_text.append(f"{data.mem_bw_util:<3}% ", style="magenta")
        footer_text.append("AI-CPU: ", style="dim")
        footer_text.append(f"{data.ai_cpu_util:<3}% ", style="blue")
        footer_text.append("Ctrl: ", style="dim")
        footer_text.append(f"{data.ctrl_cpu_util:<3}%", style="blue")
        footer_text.append("\n")
        
        # Row 2: Memory Progress Bar
        bar_width = 20
        filled = int(bar_width * mem_pct / 100)
        filled = max(0, min(filled, bar_width))
        empty = bar_width - filled
        
        bar_color = "green"
        if mem_pct > 90: bar_color = "red"
        elif mem_pct > 75: bar_color = "yellow"
        elif mem_pct > 50: bar_color = "cyan"

        footer_text.append("MEM ", style="bold white")
        footer_text.append("[", style="white")
        footer_text.append("|" * filled, style=bar_color)
        footer_text.append(" " * empty, style="dim grey23")
        footer_text.append("] ", style="white")
        footer_text.append(f"{mem_gb:.0f}/{mem_total:.0f}G", style="cyan")

        self.footer_label.update(footer_text)

        # 5. Plot
        self.plot_widget.update_plot(data.history_util, data.history_mem)

# ==========================================
# Data Fetching Logic (Optimized for Speed)
# ==========================================

class NPUReader:
    def __init__(self):
        self.static_info_cache = {}
        # Thread pool for parallel command execution
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

    def get_data(self) -> tuple[Dict[str, NPUData], List[ProcessData]]:
        npus, processes = self._fetch_standard_info()
        
        usage_futures = {}
        static_futures = {}
        
        # Submit Usage Tasks
        for nid in npus:
            future = self.executor.submit(self._fetch_usage_stats_for_id, nid)
            usage_futures[future] = nid
            
            if nid not in self.static_info_cache:
                st_future = self.executor.submit(self._fetch_static_info, nid)
                static_futures[st_future] = nid

        # Collect Usage Results
        for future in concurrent.futures.as_completed(usage_futures):
            nid = usage_futures[future]
            data = npus[nid]
            try:
                usages = future.result()
                data.ai_cpu_util = usages.get('ai_cpu', 0)
                data.ctrl_cpu_util = usages.get('ctrl_cpu', 0)
                data.mem_bw_util = usages.get('mem_bw', 0)
            except Exception:
                pass 

        # Collect Static Results
        if static_futures:
            for future in concurrent.futures.as_completed(static_futures):
                nid = static_futures[future]
                try:
                    self.static_info_cache[nid] = future.result()
                except Exception:
                    pass

        # Apply Cached Static Info
        for nid, data in npus.items():
            static_info = self.static_info_cache.get(nid, {})
            data.ai_core_count = static_info.get("ai_core_count", 0)
            data.clock_freq = static_info.get("clock_freq", 0)
            if data.hbm_total == 0: 
                data.hbm_total = static_info.get("hbm_total", 0)
                
        return npus, processes

    def _fetch_usage_stats_for_id(self, npu_id: str) -> dict:
        stats = {'ai_cpu': 0, 'ctrl_cpu': 0, 'mem_bw': 0}
        try:
            cmd = ["npu-smi", "info", "-t", "usages", "-i", str(npu_id)]
            output = subprocess.check_output(cmd, encoding="utf-8")
            
            for line in output.splitlines():
                line = line.strip()
                if ":" not in line: continue
                key_part, val_part = line.split(":", 1)
                key = key_part.strip()
                try:
                    val = int(float(val_part.strip()))
                except ValueError:
                    continue
                
                if "Aicpu Usage Rate" in key:
                    stats['ai_cpu'] = val
                elif "Ctrlcpu Usage Rate" in key:
                    stats['ctrl_cpu'] = val
                elif "HBM Bandwidth Usage Rate" in key:
                    stats['mem_bw'] = val
        except Exception:
            pass
        return stats

    def _fetch_static_info(self, npu_id: str) -> dict:
        info = {"ai_core_count": 0, "hbm_total": 0, "clock_freq": 0}
        try:
            cmd = ["npu-smi", "info", "-t", "common", "-i", str(npu_id)]
            output = subprocess.check_output(cmd, encoding="utf-8")
            for line in output.splitlines():
                line = line.strip()
                if ":" not in line: continue
                key_part, val_part = line.split(":", 1)
                key = key_part.strip()
                val_str = val_part.strip()
                
                if "Aicore Count" in key:
                    info["ai_core_count"] = int(val_str)
                elif "HBM Capacity" in key:
                    info["hbm_total"] = int(val_str)
                elif "Aicore Freq" in key or "Aicore curFreq" in key:
                    info["clock_freq"] = int(val_str)
        except Exception:
            pass
        return info

    def _get_process_cmdline(self, pid: str) -> str:
        """Attempts to read the full command line for a PID"""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                content = f.read().replace(b'\x00', b' ').strip()
                return content.decode('utf-8', errors='ignore')
        except Exception:
            return ""

    def _get_process_user(self, pid: str) -> str:
        """Attempts to get the username for a PID"""
        try:
            uid = os.stat(f"/proc/{pid}").st_uid
            return pwd.getpwuid(uid).pw_name
        except Exception:
            return "Unknown"

    def _fetch_standard_info(self) -> tuple[Dict[str, NPUData], List[ProcessData]]:
        try:
            output = subprocess.check_output(["npu-smi", "info"], encoding="utf-8")
            return self._parse_npu_smi_output(output)
        except: return {}, []

    def _parse_npu_smi_output(self, output: str) -> tuple[Dict[str, NPUData], List[ProcessData]]:
        npus = {}
        processes = []
        lines = output.splitlines()
        section = "HEADER"
        current_npu_id = None

        for line in lines:
            line = line.strip()
            if not line: continue
            if "NPU" in line and "Name" in line: section = "NPU_LIST"; continue
            if "Process id" in line: section = "PROCESS_LIST"; continue

            if section == "NPU_LIST":
                if line.startswith("+") or line.startswith("="): continue
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if not parts: continue

                first_col = parts[0].split()
                if len(first_col) >= 2 and first_col[0].isdigit():
                    npu_id = first_col[0]
                    name = first_col[1]
                    health = parts[1] if len(parts) > 1 else "Unknown"
                    pt_col = parts[2].split() if len(parts) > 2 else []
                    power = float(pt_col[0]) if len(pt_col) > 0 else 0.0
                    temp = int(pt_col[1]) if len(pt_col) > 1 else 0
                    
                    current_npu_id = npu_id
                    npus[npu_id] = NPUData(id=npu_id, name=name, health=health, power=power, temp=temp)
                
                elif len(first_col) == 1 and first_col[0].isdigit() and current_npu_id is not None:
                    bus_id = parts[1] if len(parts) > 1 else "N/A"
                    stats_col = parts[-1]
                    stats_clean = stats_col.replace("/", " ").split()
                    npu_obj = npus[current_npu_id]
                    npu_obj.bus_id = bus_id
                    if len(stats_clean) >= 5:
                        try:
                            npu_obj.ai_core_util = int(float(stats_clean[0]))
                            npu_obj.hbm_used = int(stats_clean[3])
                            npu_obj.hbm_total = int(stats_clean[4])
                        except: pass
                    current_npu_id = None

            elif section == "PROCESS_LIST":
                if line.startswith("+") or "No running" in line: continue
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 4:
                    npu_chip = parts[0].split()
                    npu_id = npu_chip[0] if npu_chip else "?"
                    pid = parts[1]
                    name = parts[2]
                    mem = float(parts[3]) if parts[3].isdigit() else 0.0
                    # Fetch cmdline and user
                    cmdline = self._get_process_cmdline(pid)
                    user = self._get_process_user(pid)
                    processes.append(ProcessData(npu_id, pid, name, mem, cmdline, user))
        return npus, processes

# ==========================================
# Main Application
# ==========================================

class NPUTopApp(App):
    CSS = """
    Screen { background: #0f0f0f; }
    
    #npu-grid {
        layout: grid;
        grid-size-columns: 4; 
        grid-rows: 1fr 1fr;
        height: 70%;
        margin: 1;
        grid-gutter: 1;
    }

    .npu-card {
        background: #1e1e1e;
        border: solid #005f5f;  /* 调整：更具科技感的深青色边框 */
        height: 100%;
        padding: 0;
    }

    .card-header { 
        text-style: bold; 
        width: 100%; 
        background: #252525;
        padding: 0 1;
    }
    
    .card-sub-header { 
        width: 100%; 
        padding: 0 1;
        color: #888888;
        border-bottom: solid #005f5f; /* 调整：匹配边框颜色 */
    }

    .plot-container {
        height: 1fr;
        width: 100%;
        padding: 0 0;
        align: center middle;
    }
    
    PolylineWidget {
        width: 100%;
        height: 100%;
    }

    .card-footer { 
        width: 100%; 
        text-align: center; 
        background: #252525; 
        border-top: solid #005f5f; /* 调整：匹配边框颜色 */
        height: 3;
    }
    
    #process-container {
        height: 30%;
        border-top: solid #005f5f; /* 调整：匹配主题色 */
        background: $surface;
    }
    DataTable { height: 100%; background: $surface; }
    """

    TITLE = "NPU-TOP Pro (Vector Edition)"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.reader = NPUReader()
        self.npu_widgets = {}
        self.npu_history_util = {}
        self.npu_history_mem = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(id="npu-grid")
        with Container(id="process-container"):
            yield Label(" Active Processes", classes="panel-header")
            yield DataTable(zebra_stripes=True)
        yield Footer()

    def on_mount(self):
        table = self.query_one(DataTable)
        # Update: Added "User" column
        table.add_columns("NPU", "PID", "User", "Process Name", "Memory (MB)", "Command")
        table.cursor_type = "row"
        self.set_interval(REFRESH_RATE, self.update_data)
        self.update_data()

    @work(exclusive=True, thread=True)
    def update_data(self):
        npu_data_map, processes = self.reader.get_data()
        self.call_from_thread(self.refresh_ui, npu_data_map, processes)

    def refresh_ui(self, npu_map: Dict[str, NPUData], processes: List[ProcessData]):
        grid = self.query_one("#npu-grid")
        sorted_ids = sorted(npu_map.keys(), key=lambda x: int(x))
        
        for nid in sorted_ids:
            data = npu_map[nid]
            
            if nid not in self.npu_history_util:
                self.npu_history_util[nid] = deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
                self.npu_history_mem[nid] = deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
            
            self.npu_history_util[nid].append(data.ai_core_util)
            mem_pct = int((data.hbm_used / data.hbm_total * 100)) if data.hbm_total else 0
            self.npu_history_mem[nid].append(mem_pct)
            
            data.history_util = self.npu_history_util[nid]
            data.history_mem = self.npu_history_mem[nid]

            if nid not in self.npu_widgets:
                widget = NPUCard(nid)
                grid.mount(widget)
                self.npu_widgets[nid] = widget
            
            self.npu_widgets[nid].update(data)

        table = self.query_one(DataTable)
        table.clear()
        for p in processes:
            # Update: Added p.user to the row
            table.add_row(p.npu_id, p.pid, p.user, p.name, f"{p.memory:.1f}", p.cmdline)

if __name__ == "__main__":
    app = NPUTopApp()
    app.run()