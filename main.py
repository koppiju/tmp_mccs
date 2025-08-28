# -*- coding: utf-8 -*-
import curses
import subprocess
import re
from typing import Dict, Any, List, Optional, Tuple

# ---------------- JSON helpers (with list support) ----------------

def is_dict(x: Any) -> bool: return isinstance(x, dict)
def is_list(x: Any) -> bool: return isinstance(x, list)
def is_container(x: Any) -> bool: return is_dict(x) or is_list(x)

def idx_token(i: int) -> str: return f"[{i}]"

def parse_idx_token(tok: str) -> Optional[int]:
    if isinstance(tok, str) and len(tok) >= 3 and tok[0] == "[" and tok[-1] == "]":
        try: return int(tok[1:-1])
        except Exception: return None
    return None

def get_child(node: Any, token: str) -> Tuple[bool, Any]:
    if is_dict(node):
        if token in node: return True, node[token]
        return False, None
    if is_list(node):
        i = parse_idx_token(token)
        if i is not None and 0 <= i < len(node): return True, node[i]
        return False, None
    return False, None

def parse_like(original: Any, s: str) -> Any:
    if isinstance(original, bool):
        sl = s.strip().lower()
        if sl in ("true", "1", "yes", "on"): return True
        if sl in ("false", "0", "no", "off"): return False
        return bool(s)
    if isinstance(original, int) and not isinstance(original, bool):
        try: return int(s.strip())
        except: return s
    if isinstance(original, float):
        try: return float(s.strip())
        except: return s
    return s

def get_in(root: Any, tokens: List[str]) -> Any:
    cur = root
    for t in tokens:
        ok, cur = get_child(cur, t)
        if not ok: return None
    return cur

def set_in(root: Any, tokens: List[str], value: Any) -> bool:
    if not tokens: return False
    if len(tokens) == 1:
        parent = root; last = tokens[0]
    else:
        parent = get_in(root, tokens[:-1]); last = tokens[-1]
    if is_dict(parent):
        parent[last] = value; return True
    if is_list(parent):
        i = parse_idx_token(last)
        if i is not None and 0 <= i < len(parent):
            parent[i] = value; return True
    return False

# ---------------- Enum helpers ----------------

HARD_CODED_ENUM_VARIANTS = ["Status1", "Status2", "Status3", "Status4"]

def enum_parts(val: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(val, str) or not val.startswith("enum::"): return None
    rest = val[6:]
    if "::" not in rest: return None
    type_name, value_name = rest.split("::", 1)
    if not type_name or not value_name: return None
    return type_name, value_name

def enum_variants_for(type_name: str) -> List[str]:
    return HARD_CODED_ENUM_VARIANTS[:]

# ---------------- History helpers ----------------

class HistItem:
    def __init__(self, idx: int, timestamp: str, header_tail: str):
        self.idx = idx
        self.timestamp = timestamp
        self.header_tail = header_tail
        self.changes: List[str] = []

def run_mccs_history(config: str, instance: str, root_path: str) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["mccs", "history", config, instance, root_path],
            capture_output=True, text=True, check=False
        )
        if proc.returncode == 0:
            return True, proc.stdout
        else:
            return False, proc.stderr.strip() or f"mccs history failed with code {proc.returncode}"
    except FileNotFoundError:
        return False, "mccs not found in PATH"
    except Exception as e:
        return False, f"error running mccs: {e}"

def run_mccs_pull(config: str, instance: str, version: int) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["mccs", "pull", config, instance, str(version)],
            capture_output=True, text=True, check=False
        )
        if proc.returncode == 0:
            return True, proc.stdout
        else:
            return False, proc.stderr.strip() or f"mccs pull failed with code {proc.returncode}"
    except FileNotFoundError:
        return False, "mccs not found in PATH"
    except Exception as e:
        return False, f"error running mccs: {e}"

HEADER_RE = re.compile(r"^\[(\d+)\]\[(.*?)\]:\s*(.*)$")
ANSI_RE = re.compile(r"\x1b\[((?:\d{1,3};?)+)m")  # used below as well

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def parse_history_output(text: str) -> List['HistItem']:
    """
    Parse history, capturing headers and change-lines.
    Change-lines may be colorized; we strip ANSI only for detection but keep
    the original text (with ANSI) for rendering.
    """
    items: List[HistItem] = []
    current: Optional[HistItem] = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        m = HEADER_RE.match(strip_ansi(line))
        if m:
            idx = int(m.group(1)); ts = m.group(2); tail = m.group(3)
            # keep original tail (with any ANSI) if present after ']: '
            if "]: " in line:
                tail_disp = line[line.find("]: ") + 3 :]
            else:
                tail_disp = tail
            current = HistItem(idx, ts, tail_disp)
            items.append(current)
            continue
        if current is None:
            continue
        trimmed = line.lstrip()
        if not trimmed:
            continue
        det = strip_ansi(trimmed).lstrip()
        if det and det[0] in ("u", "U", "+", "-"):
            current.changes.append(trimmed)
    return items

# ---------------- TUI ----------------

def run_json_tui(data: Dict[str, Any], startup_alert: Optional[str] = None) -> Dict[str, Any]:
    """
    Three rounded boxes (full width):
      Box 1: path (1 line)
      Box 2: navigator (1 header + 10 rows + 1 help = 12 inner rows)
      Box 3: bottom (menu row + content + help)
    """
    N_COLS = 5

    # ------------- State -------------
    cursor = [0] * N_COLS
    locked_idx = [None] * N_COLS
    locked_by_depth: Dict[int, str] = {}
    counter = 0
    focus = 0

    alert_open = bool(startup_alert)
    alert_message = startup_alert or ""

    MENU_HISTORY, MENU_CHANGES, MENU_LOGS = 0, 1, 2
    menu_selected = MENU_HISTORY

    lists: List[List[str]] = [[] for _ in range(N_COLS)]
    leaf_info: Optional[Tuple[int, str, Any]] = None

    # input editor
    input_mode = False
    input_text = ""
    input_cursor = 0
    input_scroll = 0
    input_target_path: Optional[List[str]] = None
    input_original_value: Any = None
    input_leaf_key: Optional[str] = None
    input_parent_chain: List[str] = []
    input_saved_cursor: Optional[List[int]] = None
    input_saved_counter: Optional[int] = None
    input_return_focus: Optional[int] = None
    input_caret_yx: Optional[Tuple[int, int]] = None

    # bool editor
    bool_mode = False
    bool_cursor = 0
    bool_target_path: Optional[List[str]] = None
    bool_original_value: Optional[bool] = None
    bool_saved_cursor: Optional[List[int]] = None
    bool_saved_counter: Optional[int] = None
    bool_return_focus: Optional[int] = None

    # enum editor
    enum_mode = False
    enum_cursor = 0
    enum_variants: List[str] = []
    enum_type_name: Optional[str] = None
    enum_target_path: Optional[List[str]] = None
    enum_original_value: Optional[str] = None
    enum_saved_cursor: Optional[List[int]] = None
    enum_saved_counter: Optional[int] = None
    enum_return_focus: Optional[int] = None

    # find mode
    find_mode = False
    find_query = ""
    find_cursor = 0
    find_scroll_x = 0
    find_results_all: List[Tuple[List[str], str]] = []
    find_results: List[Tuple[List[str], str]] = []
    find_sel = 0
    find_list_scroll = 0
    find_input_yx: Optional[Tuple[int, int]] = None
    find_mem_locked_by_depth: Optional[Dict[int, str]] = None
    find_mem_counter: Optional[int] = None
    find_mem_cursor: Optional[List[int]] = None
    find_mem_focus: Optional[int] = None
    find_mem_locked_idx: Optional[List[Optional[int]]] = None

    # history
    hist_entries: List[HistItem] = []
    hist_sel = 0
    hist_scroll = 0
    # what we actually show right now
    hist_show_changes: bool = False
    # user's override (None = use auto-fit)
    hist_user_forced_show: Optional[bool] = None
    hist_last_query_key: Optional[Tuple[str, str, str]] = None
    hist_error: Optional[str] = None

    prev_eff_cfg: Optional[str] = None
    prev_eff_inst: Optional[str] = None

    # ------------- Helpers -------------

    def list_items(node: Any) -> List[str]:
        if is_dict(node): return list(node.keys())
        if is_list(node): return [idx_token(i) for i in range(len(node))]
        return []

    def effective_cfg_key() -> Optional[str]:
        col0 = list(data.keys())
        if not col0: return None
        idx = locked_idx[0] if locked_idx[0] is not None else cursor[0]
        idx = max(0, min(idx, len(col0) - 1))
        return col0[idx]

    def effective_inst_key(cfg_key: Optional[str]) -> Optional[str]:
        if cfg_key is None: return None
        node = data.get(cfg_key)
        col1 = list_items(node)
        if not col1: return None
        idx = locked_idx[1] if locked_idx[1] is not None else cursor[1]
        idx = max(0, min(idx, len(col1) - 1))
        return col1[idx]

    def base_node(cfg_key: Optional[str], inst_key: Optional[str]) -> Any:
        if cfg_key is None or inst_key is None: return None
        ok, node = get_child(data.get(cfg_key, {}), inst_key)
        return node if ok else None

    def left_depth() -> int:  return 3 + counter
    def right_depth() -> int: return 5 + counter

    def depth_to_col(d: int) -> Optional[int]:
        l = left_depth(); r = right_depth()
        if l <= d <= r:
            return 2 + (d - l)
        return None

    def token_at_depth(d: int) -> Optional[str]:
        if d in locked_by_depth: return locked_by_depth[d]
        col = depth_to_col(d)
        if col is None: return None
        if 0 <= col < len(lists) and lists[col]:
            i = max(0, min(cursor[col], len(lists[col]) - 1))
            return lists[col][i]
        return None

    def parent_node_for_depth(parent_depth: int,
                              cfg_key: Optional[str],
                              inst_key: Optional[str]) -> Optional[Any]:
        if cfg_key is None or inst_key is None: return None
        node = base_node(cfg_key, inst_key)
        if parent_depth <= 2: return node
        for d in range(3, parent_depth + 1):
            if not is_container(node): return None
            tok = token_at_depth(d)
            if not tok: return None
            ok, node = get_child(node, tok)
            if not ok: return None
        return node

    def prune_inconsistent_locks(cfg_key: Optional[str], inst_key: Optional[str]) -> None:
        node = base_node(cfg_key, inst_key); d = 3
        while True:
            if not is_container(node):
                for dd in list(locked_by_depth.keys()):
                    if dd >= d: locked_by_depth.pop(dd, None)
                return
            tok = locked_by_depth.get(d)
            if tok is None:
                for dd in list(locked_by_depth.keys()):
                    if dd > d: locked_by_depth.pop(dd, None)
                return
            ok, node = get_child(node, tok)
            if not ok:
                for dd in list(locked_by_depth.keys()):
                    if dd >= d: locked_by_depth.pop(dd, None)
                return
            d += 1

    def draw_centered_alert_in_nav(stdscr, nav_y0: int, nav_y1: int, w: int, msg: str):
        # Compute a centered  box within the navigator box (rows [nav_y0..nav_y1])
        lines = [ln for ln in msg.splitlines()]
        inner_w = min(w - 6, max(20, max((len(ln) for ln in lines), default=20)))
        inner_h = len(lines) + 2  # padding
        box_w = inner_w + 2
        box_h = inner_h + 2

        cy = nav_y0 + max(0, (nav_y1 - nav_y0 + 1 - box_h) // 2)
        cx = max(0, (w - box_w) // 2)

        # Rounded box
        top = "╭" + "─" * inner_w + "╮"
        mid = "│" + " " * inner_w + "│"
        bot = "╰" + "─" * inner_w + "╯"
        stdscr.addnstr(cy,     cx, top,   box_w)
        for i in range(inner_h):
            stdscr.addnstr(cy+1+i, cx, mid, box_w)
        stdscr.addnstr(cy+1+inner_h, cx, bot,   box_w)

        # Text
        ty = cy + 2 - 1  # one line padding
        for i, ln in enumerate(lines):
            ln = ln[:inner_w]
            stdscr.addnstr(ty + i, cx + 1, ln.ljust(inner_w), inner_w, curses.A_BOLD)


    # ---- Build lists & sync cursors ----

    def build_lists() -> Tuple[List[List[str]], Optional[Tuple[int, str, Any]]]:
        the_lists = [[] for _ in range(N_COLS)]
        the_lists[0] = list(data.keys())
        eff_cfg = effective_cfg_key()
        if eff_cfg is not None:
            the_lists[1] = list_items(data.get(eff_cfg))
        eff_inst = effective_inst_key(eff_cfg)

        for col in (2, 3, 4):
            d_target = left_depth() + (col - 2)
            parent = parent_node_for_depth(d_target - 1, eff_cfg, eff_inst)
            if not is_container(parent): the_lists[col] = []
            else: the_lists[col] = list_items(parent)

        # leaf detection for previews
        leaf: Optional[Tuple[int, str, Any]] = None
        if the_lists[2]:
            parent2 = parent_node_for_depth(left_depth()-1, eff_cfg, eff_inst)
            tok2 = token_at_depth(left_depth())
            if tok2 and is_container(parent2):
                ok2, v2 = get_child(parent2, tok2)
                if ok2 and not is_container(v2): leaf = (2, tok2, v2)
        if leaf is None and the_lists[3]:
            parent3 = parent_node_for_depth(left_depth(), eff_cfg, eff_inst)
            tok3 = token_at_depth(left_depth()+1)
            if tok3 and is_container(parent3):
                ok3, v3 = get_child(parent3, tok3)
                if ok3 and not is_container(v3): leaf = (3, tok3, v3)

        if input_mode and input_leaf_key: the_lists[2] = [input_leaf_key]
        if bool_mode  and bool_target_path: the_lists[2] = [bool_target_path[-1]]
        if enum_mode  and enum_target_path: the_lists[2] = [enum_target_path[-1]]
        return the_lists, leaf

    def sync_cursors_to_locks(the_lists: List[List[str]]) -> None:
        for col in (2, 3, 4):
            d = left_depth() + (col - 2)
            tok = locked_by_depth.get(d); n = len(the_lists[col])
            if n == 0: cursor[col] = 0; continue
            if tok is not None:
                try: cursor[col] = the_lists[col].index(tok)
                except ValueError: cursor[col] = 0
            else:
                cursor[col] = max(0, min(cursor[col], n - 1))

    def set_cursor_for_depth_token(depth: int, token: Optional[str]):
        if not token: return
        col = depth_to_col(depth)
        if col is None: return
        lst = lists[col]
        if not lst: return
        try: cursor[col] = lst.index(token)
        except ValueError: pass

    # ---------------- Effective path ----------------

    def build_snapshot_path() -> List[str]:
        parts: List[str] = []
        eff_cfg = effective_cfg_key(); eff_inst = effective_inst_key(eff_cfg)
        if eff_cfg: parts.append(eff_cfg)
        if eff_inst: parts.append(eff_inst)
        node = base_node(eff_cfg, eff_inst); dcur = 3
        while True:
            if not is_container(node) or dcur > left_depth(): break
            tok = locked_by_depth.get(dcur)
            if not tok: break
            ok, node = get_child(node, tok)
            if not ok: break
            parts.append(tok); dcur += 1
        if input_mode and input_leaf_key: parts.append(input_leaf_key)
        elif bool_mode and bool_target_path: parts.append(bool_target_path[-1])
        elif enum_mode and enum_target_path: parts.append(enum_target_path[-1])
        return parts

    def compute_effective_path(the_lists: List[List[str]]) -> List[str]:
        if input_mode or bool_mode or enum_mode or find_mode:
            return build_snapshot_path()
        parts: List[str] = []
        if the_lists[0]:
            parts.append(the_lists[0][max(0, min(cursor[0], len(the_lists[0])-1))])
        eff_cfg_locked = (locked_idx[0] is not None)
        if eff_cfg_locked or focus >= 1:
            if the_lists[1]:
                parts.append(the_lists[1][max(0, min(cursor[1], len(the_lists[1])-1))])
        eff_cfg = effective_cfg_key(); eff_inst = effective_inst_key(eff_cfg)
        node = base_node(eff_cfg, eff_inst); d = 3
        while True:
            if not is_container(node): break
            tok = locked_by_depth.get(d)
            if not tok: break
            ok, node = get_child(node, tok)
            if not ok: break
            parts.append(tok); d += 1
        if focus in (2, 3, 4):
            dfocus = left_depth() + (focus - 2)
            curd = d
            while curd <= dfocus:
                col = depth_to_col(curd)
                if col is None or not the_lists[col]: break
                parts.append(the_lists[col][max(0, min(cursor[col], len(the_lists[col])-1))])
                curd += 1
        return parts

    # ---------- Borders & drawing basics ----------

    def draw_box(stdscr, y0: int, y1: int, w: int, rounded: bool = True):
        h, sw = stdscr.getmaxyx()
        y0 = max(0, min(y0, h - 1)); y1 = max(0, min(y1, h - 1))
        if y1 <= y0 or w <= 2: return
        safe_w = min(w, sw) - 1
        if safe_w < 2: return
        inner = max(0, safe_w - 2)
        tl, tr, bl, br = ("╭","╮","╰","╯") if rounded else ("┌","┐","└","┘")
        stdscr.addnstr(y0, 0, tl + "─" * inner + tr, safe_w)
        for y in range(y0 + 1, y1):
            stdscr.addnstr(y, 0, "│", 1)
            stdscr.addnstr(y, 1, " " * inner, inner)
            stdscr.addnstr(y, safe_w - 1, "│", 1)
        stdscr.addnstr(y1, 0, bl + "─" * inner + br, safe_w)

    def draw_centered(stdscr, y0: int, y1: int, x0: int, w: int, msg: str, attr=curses.A_DIM):
        rows = max(1, y1 - y0 + 1)
        y = y0 + rows // 2
        x = x0 + max(0, (w - len(msg)) // 2)
        stdscr.addnstr(y, x, msg[:w], w, attr)

    # ---- ANSI support for history rendering ----
    def _init_ansi_pairs():
        base_codes = [curses.COLOR_BLACK, curses.COLOR_RED, curses.COLOR_GREEN, curses.COLOR_YELLOW,
                      curses.COLOR_BLUE, curses.COLOR_MAGENTA, curses.COLOR_CYAN, curses.COLOR_WHITE]
        pair_base = 10
        pairs = {}
        for i, fg in enumerate(base_codes):
            try: curses.init_pair(pair_base + i, fg, -1)
            except curses.error: curses.init_pair(pair_base + i, curses.COLOR_WHITE, -1)
            pairs[30 + i] = pair_base + i
            pairs[90 + i] = pair_base + i
        return pairs

    def draw_ansi(stdscr, y: int, x: int, text: str, max_w: int, default_attr=curses.A_NORMAL, ansi_pairs=None):
        if ansi_pairs is None: ansi_pairs = {}
        cur_attr = default_attr; cur_pair = 0
        pos = 0; written = 0
        for m in ANSI_RE.finditer(text):
            literal = text[pos:m.start()]
            if literal:
                take = min(len(literal), max(0, max_w - written))
                if take > 0:
                    stdscr.addnstr(y, x + written, literal[:take], take, cur_attr | curses.color_pair(cur_pair))
                    written += take
                    if written >= max_w: return
            codes = [c for c in m.group(1).split(";") if c]
            for c in codes:
                try: n = int(c)
                except: continue
                if n == 0:
                    cur_attr = default_attr; cur_pair = 0
                elif n == 1:
                    cur_attr |= curses.A_BOLD
                elif n == 39:
                    cur_pair = 0
                elif 30 <= n <= 37 or 90 <= n <= 97:
                    cur_pair = ansi_pairs.get(n, 0)
            pos = m.end()
        tail = text[pos:]
        if tail and written < max_w:
            take = min(len(tail), max(0, max_w - written))
            if take > 0:
                stdscr.addnstr(y, x + written, tail[:take], take, cur_attr | curses.color_pair(cur_pair))

    # --- FIND helpers/drawing ---

    def format_display_path(cfg: str, inst: str, tokens: List[str]) -> str:
        s = f"${cfg}::{inst}"
        for t in tokens:
            s += (t if t.startswith("[") else f".{t}")
        return s

    def collect_leaves(node: Any, prefix_tokens: List[str], out: List[List[str]]) -> None:
        if is_dict(node):
            for k in list(node.keys()):
                ok, v = get_child(node, k)
                if not ok: continue
                if is_container(v): collect_leaves(v, prefix_tokens + [k], out)
                else: out.append(prefix_tokens + [k])
        elif is_list(node):
            for i in range(len(node)):
                tok = idx_token(i)
                ok, v = get_child(node, tok)
                if not ok: continue
                if is_container(v): collect_leaves(v, prefix_tokens + [tok], out)
                else: out.append(prefix_tokens + [tok])

    def rebuild_find_index(cfg: Optional[str], inst: Optional[str]) -> None:
        nonlocal find_results_all
        find_results_all = []
        if not cfg or not inst: return
        node = base_node(cfg, inst)
        if not is_container(node): return
        paths: List[List[str]] = []
        collect_leaves(node, [], paths)
        for toks in paths:
            disp = format_display_path(cfg, inst, toks)
            find_results_all.append((toks, disp))

    def filter_find_results() -> None:
        nonlocal find_results, find_sel, find_list_scroll
        q = find_query.strip().lower()
        if not q: find_results = find_results_all[:]
        else: find_results = [(t, d) for (t, d) in find_results_all if q in d.lower()]
        if not find_results:
            find_sel = 0; find_list_scroll = 0
        else:
            find_sel = max(0, min(find_sel, len(find_results) - 1))

    def draw_find_path(stdscr, y, x0, w, cfg: Optional[str], inst: Optional[str]) -> None:
        nonlocal find_input_yx, find_scroll_x
        stdscr.addnstr(y, x0, " " * max(0, w), max(0, w))
        x = x0
        stdscr.addnstr(y, x, "$", 1, curses.A_BOLD); x += 1
        if cfg:
            stdscr.addnstr(y, x, cfg[: max(0, x0 + w - x)], min(len(cfg), max(0, x0 + w - x)), curses.A_BOLD | curses.color_pair(1)); x += len(cfg)
        if x < x0 + w:
            stdscr.addnstr(y, x, "::", min(2, max(0, x0 + w - x)), curses.A_DIM); x += 2
        if inst and x < x0 + w:
            stdscr.addnstr(y, x, inst[: max(0, x0 + w - x)], min(len(inst), max(0, x0 + w - x)), curses.A_BOLD | curses.color_pair(2)); x += len(inst)
        if x < x0 + w:
            stdscr.addnstr(y, x, ": ", min(2, max(0, x0 + w - x))); x += 2

        inner_w = max(1, x0 + w - x)
        if find_cursor < find_scroll_x: find_scroll_x = find_cursor
        elif find_cursor > find_scroll_x + inner_w - 1: find_scroll_x = find_cursor - inner_w + 1
        view = find_query[find_scroll_x: find_scroll_x + inner_w]
        stdscr.addnstr(y, x, view.ljust(inner_w), inner_w)
        find_input_yx = (y, x + (find_cursor - find_scroll_x))

    def draw_find_results_list(stdscr, y0: int, y1: int, x0: int, w: int) -> None:
        nonlocal find_list_scroll, find_sel
        safe_w = max(1, w); rows = max(0, y1 - y0 + 1); n = len(find_results)
        if rows <= 0: return
        if n == 0:
            find_list_scroll = 0
            stdscr.addnstr(y0, x0, "— no matches —"[:safe_w], safe_w, curses.A_DIM)
            for y in range(y0 + 1, y1 + 1): stdscr.addnstr(y, x0, " " * safe_w, safe_w)
            return
        if find_sel < find_list_scroll: find_list_scroll = find_sel
        elif find_sel >= find_list_scroll + rows: find_list_scroll = find_sel - rows + 1
        find_list_scroll = max(0, min(find_list_scroll, max(0, n - rows)))
        for row in range(rows):
            i = find_list_scroll + row; y = y0 + row
            if i >= n:
                stdscr.addnstr(y, x0, " " * safe_w, safe_w); continue
            _, disp = find_results[i]; is_sel = (i == find_sel)
            prefix = "> " if is_sel else "  "
            cell_w = max(1, safe_w - len(prefix))
            text_disp = (disp[:cell_w - 1] + "…") if len(disp) > cell_w else disp.ljust(cell_w)
            attrs = curses.A_REVERSE | curses.A_BOLD if is_sel else curses.A_NORMAL
            if is_sel: stdscr.addnstr(y, x0, " " * safe_w, safe_w, curses.A_REVERSE)
            stdscr.addnstr(y, x0, prefix + text_disp, safe_w, attrs)

    # --- Preview navigator for a leaf path (cfg/inst/toks) ---

    def build_preview_navigator(cfg: str, inst: str, toks: List[str]) -> Tuple[List[List[str]], List[int], Optional[Tuple[int, Any]]]:
        pv_lists = [[] for _ in range(N_COLS)]
        pv_lists[0] = list(data.keys())
        pv_lists[1] = list_items(data.get(cfg, {}))
        node = base_node(cfg, inst)
        chain_nodes = [node]; cur = node
        for t in toks:
            ok, nxt = get_child(cur, t)
            if not ok: break
            chain_nodes.append(nxt); cur = nxt
        L_abs = 2 + len(toks)
        pv_counter = max(0, L_abs - 4); left = 3 + pv_counter
        for col in (2, 3, 4):
            d_target = left + (col - 2); parent_depth = d_target - 1
            idx_in_chain = parent_depth - 2
            if 0 <= idx_in_chain < len(chain_nodes):
                parent = chain_nodes[idx_in_chain]
                pv_lists[col] = list_items(parent) if is_container(parent) else []
            else:
                pv_lists[col] = []
        pv_cursors = [0, 0, 0, 0, 0]
        try: pv_cursors[0] = pv_lists[0].index(cfg)
        except ValueError: pv_cursors[0] = 0
        try: pv_cursors[1] = pv_lists[1].index(inst) if pv_lists[1] else 0
        except ValueError: pv_cursors[1] = 0
        for col in (2, 3, 4):
            d = left + (col - 2); idx_tok = d - 3
            if 0 <= idx_tok < len(toks):
                token = toks[idx_tok]
                if pv_lists[col]:
                    try: pv_cursors[col] = pv_lists[col].index(token)
                    except ValueError: pv_cursors[col] = 0
            else:
                pv_cursors[col] = 0
        leaf_col = None
        if L_abs == left: leaf_col = 2
        elif L_abs == left + 1: leaf_col = 3
        elif L_abs == left + 2: leaf_col = 4
        leaf_preview = None
        if leaf_col in (2, 3):
            val = get_in(chain_nodes[0], toks) if chain_nodes else None
            leaf_preview = (leaf_col, val)
        return pv_lists, pv_cursors, leaf_preview

    # --- Bottom menu ---

    def draw_bottom_menu(stdscr, y, x0, w, selected: int):
        items = ["h history", "c active changes", "l logs"]
        sep = " │ "; inside = sep.join(items)
        cx = x0 + max(0, (w - len(inside)) // 2)
        for i, label in enumerate(items):
            attr = curses.A_REVERSE | curses.A_BOLD if i == selected else curses.A_DIM
            stdscr.addnstr(y, cx, label, len(label), attr)
            cx += len(label)
            if i < len(items) - 1:
                stdscr.addnstr(y, cx, sep, len(sep), curses.A_DIM); cx += len(sep)

    # ------------- Column & editor drawing -------------

    def draw_column(stdscr, x0, width, top, height,
                    items, col_idx, is_focused, cur_idx, locked_mark_idx,
                    color_pair, counter_show=None, force_focus_idx: Optional[int] = None,
                    suppress_empty_placeholder: bool = False):
        title = ""
        if col_idx == 0: title = "modules"
        if col_idx == 1: title = "instance"
        if col_idx == 3 and (counter_show is not None):
            title = f"{title} ({counter_show})" if title else f"({counter_show})"
        attr_title = curses.A_BOLD | curses.color_pair(color_pair)
        if is_focused: attr_title |= curses.A_UNDERLINE
        stdscr.addnstr(top - 1, x0, title[:max(0,width)], max(0, width), attr_title)

        n = len(items); cur_idx = 0 if n == 0 else max(0, min(cur_idx, n - 1))
        visible = height; scroll = 0
        if n > visible:
            if cur_idx < scroll: scroll = cur_idx
            if cur_idx >= scroll + visible: scroll = cur_idx - visible + 1
            scroll = max(0, min(scroll, n - visible))

        for row in range(visible):
            y = top + row; idx = scroll + row
            if n == 0:
                if suppress_empty_placeholder: continue
                text = "— no options —" if row == 0 else ""
                cursor_here = (row == 0) and is_focused; locked_here = False
            else:
                if idx >= n:
                    text = ""; cursor_here = False; locked_here = False
                else:
                    text = items[idx]; cursor_here = is_focused and (idx == cur_idx)
                    if (force_focus_idx is not None) and (idx == force_focus_idx): cursor_here = True
                    locked_here = (locked_mark_idx is not None and idx == locked_mark_idx)
            prefix = "> " if cursor_here else "  "
            cell_w = max(1, width - len(prefix))
            text_disp = (text[:cell_w - 1] + "…") if len(text) > cell_w else text.ljust(cell_w)
            attrs = curses.color_pair(color_pair)
            if cursor_here: attrs |= curses.A_REVERSE | curses.A_BOLD
            elif locked_here:
                attrs |= curses.A_REVERSE if col_idx in (0, 1) else curses.A_BOLD
            stdscr.addnstr(y, x0, prefix + text_disp, width, attrs)

    def draw_value_field(stdscr, x0, width, top, height, value_text: str):
        stdscr.addnstr(top - 1, x0, "value", max(0, width), curses.A_BOLD)
        line_y = top; prefix = "= "; inner_w = max(1, width - len(prefix))
        view = ("" if value_text is None else str(value_text))[:inner_w]
        stdscr.addnstr(line_y, x0, " " * width, width)
        stdscr.addnstr(line_y, x0, prefix + view.ljust(inner_w), width)

    def draw_input(stdscr, x0, width, top, height):
        nonlocal input_scroll, input_caret_yx
        stdscr.addnstr(top - 1, x0, "input", max(0, width), curses.A_BOLD)
        line_y = top; prefix = "= "; inner_w = max(1, width - len(prefix))
        if input_cursor < input_scroll: input_scroll = input_cursor
        elif input_cursor > input_scroll + inner_w - 1: input_scroll = input_cursor - inner_w + 1
        view = input_text[input_scroll: input_scroll + inner_w]
        stdscr.addnstr(line_y, x0, " " * width, width)
        stdscr.addnstr(line_y, x0, prefix + view.ljust(inner_w), width)
        input_caret_yx = (line_y, x0 + len(prefix) + (input_cursor - input_scroll))

    def draw_bool_editor(stdscr, x0, width, top, height, focused: bool, current_index: int):
        items = ["true", "false"]; title = "bool"
        attr_title = curses.A_BOLD | curses.color_pair(5)
        if focused: attr_title |= curses.A_UNDERLINE
        stdscr.addnstr(top - 1, x0, title, max(0, width), attr_title)
        rows_for_items = max(1, height - 1); n = len(items); rows_drawn = min(rows_for_items, n)
        start = 0
        if bool_cursor < start: start = bool_cursor
        if bool_cursor >= start + rows_drawn: start = bool_cursor - rows_drawn + 1
        start = max(0, min(start, max(0, n - rows_drawn)))
        for row in range(rows_drawn):
            i = start + row; y = top + row; text = items[i]
            cursor_here = (i == bool_cursor) and focused; is_current = (i == current_index)
            prefix = "= " if is_current else "  "; cell_w = max(1, width - len(prefix))
            text_disp = (text[:cell_w - 1] + "…") if len(text) > cell_w else text.ljust(cell_w)
            attrs = curses.color_pair(5)
            if cursor_here: attrs |= curses.A_REVERSE | curses.A_BOLD
            stdscr.addnstr(y, x0, prefix + text_disp, width, attrs)

    def draw_enum_editor(stdscr, x0, width, top, height, focused: bool,
                         items: List[str], type_name: Optional[str], current_index: Optional[int]):
        title = type_name or "enum"; attr_title = curses.A_BOLD | curses.color_pair(5)
        if focused: attr_title |= curses.A_UNDERLINE
        stdscr.addnstr(top - 1, x0, title, max(0, width), attr_title)
        n = len(items); rows_for_items = max(1, height - 1); rows_drawn = min(rows_for_items, n)
        start = 0
        if enum_cursor < start: start = enum_cursor
        if enum_cursor >= start + rows_drawn: start = enum_cursor - rows_drawn + 1
        start = max(0, min(start, max(0, n - rows_drawn)))
        for row in range(rows_drawn):
            i = start + row; y = top + row; text = items[i]
            cursor_here = (i == enum_cursor) and focused
            is_current = (current_index is not None and i == current_index)
            prefix = "= " if is_current else "  "; cell_w = max(1, width - len(prefix))
            text_disp = (text[:cell_w - 1] + "…") if len(text) > cell_w else text.ljust(cell_w)
            attrs = curses.color_pair(5)
            if cursor_here: attrs |= curses.A_REVERSE | curses.A_BOLD
            stdscr.addnstr(y, x0, prefix + text_disp, width, attrs)

    # ---------------- History drawing ----------------

    def build_root_path_for_history(tokens: List[str]) -> str:
        s = "root"
        for t in tokens:
            s += t if t.startswith("[") else f".{t}"
        return s

    def entry_height(i: int, show_changes: bool) -> int:
        if 0 <= i < len(hist_entries):
            return 1 + (len(hist_entries[i].changes) if show_changes else 0)
        return 1

    def sum_heights(i0: int, i1: int, show_changes: bool) -> int:
        if i0 > i1: return 0
        total = 0
        for i in range(i0, i1 + 1): total += entry_height(i, show_changes)
        return total

    def ensure_sel_visible(visible_rows: int, show_changes: bool):
        nonlocal hist_scroll
        if not hist_entries: hist_scroll = 0; return
        hist_sel_clamped = max(0, min(hist_sel, len(hist_entries) - 1))
        if hist_sel_clamped < hist_scroll:
            hist_scroll = hist_sel_clamped; return
        while sum_heights(hist_scroll, hist_sel_clamped, show_changes) > visible_rows:
            hist_scroll += 1
            if hist_scroll > hist_sel_clamped:
                hist_scroll = hist_sel_clamped; break

    def draw_history_list(stdscr, y0: int, y1: int, x0: int, w: int, ansi_pairs: Dict[int, int], show_changes: bool):
        rows = max(0, y1 - y0 + 1)
        for y in range(y0, y1 + 1): stdscr.addnstr(y, x0, " " * w, w)
        if hist_error:
            draw_centered(stdscr, y0, y1, x0, w, f"(history) {hist_error}", curses.A_DIM | curses.A_BOLD); return
        if not hist_entries:
            draw_centered(stdscr, y0, y1, x0, w, "no history for this path.", curses.A_DIM); return
        y = y0; i = hist_scroll
        while y <= y1 and i < len(hist_entries):
            it = hist_entries[i]
            header = f"[{it.idx}][{it.timestamp}]: {it.header_tail}"
            is_sel = (i == hist_sel)
            prefix = "> " if is_sel else "  "
            cell_w = max(1, w - len(prefix))
            if is_sel: stdscr.addnstr(y, x0, " " * w, w, curses.A_REVERSE)
            draw_ansi(stdscr, y, x0, prefix, len(prefix),
                      curses.A_REVERSE | curses.A_BOLD if is_sel else curses.A_NORMAL, ansi_pairs)
            draw_ansi(stdscr, y, x0 + len(prefix), header[:cell_w], cell_w,
                      (curses.A_REVERSE | curses.A_BOLD) if is_sel else curses.A_NORMAL, ansi_pairs)
            y += 1
            if y > y1: break
            if show_changes and it.changes:
                for chline in it.changes:
                    if y > y1: break
                    draw_ansi(stdscr, y, x0, "  ", 2, curses.A_DIM, ansi_pairs)
                    draw_ansi(stdscr, y, x0 + 2, chline[:w-2], w-2, curses.A_DIM, ansi_pairs)
                    y += 1
            i += 1

    # ---------------- App loop ----------------

    def main(stdscr):
        nonlocal focus, counter, input_mode, input_text, input_cursor, input_scroll
        nonlocal input_target_path, input_original_value, input_leaf_key, input_parent_chain
        nonlocal input_saved_cursor, input_saved_counter, input_return_focus, input_caret_yx
        nonlocal prev_eff_cfg, prev_eff_inst, locked_by_depth, lists, leaf_info
        nonlocal bool_mode, bool_cursor, bool_target_path, bool_original_value
        nonlocal bool_saved_cursor, bool_saved_counter, bool_return_focus
        nonlocal enum_mode, enum_cursor, enum_variants, enum_type_name, enum_target_path
        nonlocal enum_original_value, enum_saved_cursor, enum_saved_counter, enum_return_focus
        nonlocal menu_selected
        nonlocal find_mode, find_query, find_cursor, find_scroll_x
        nonlocal find_results_all, find_results, find_sel, find_list_scroll, find_input_yx
        nonlocal find_mem_locked_by_depth, find_mem_counter, find_mem_cursor, find_mem_focus, find_mem_locked_idx
        nonlocal hist_entries, hist_sel, hist_scroll, hist_show_changes, hist_last_query_key, hist_error, hist_user_forced_show

        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        try: curses.set_escdelay(25)
        except Exception: pass

        has_colors = curses.has_colors()
        if has_colors:
            curses.start_color()
            try: curses.use_default_colors()
            except curses.error: pass
            ORANGE = 208 if getattr(curses, "COLORS", 8) >= 256 else curses.COLOR_MAGENTA
            GREY = 244 if getattr(curses, "COLORS", 8) >= 256 else curses.COLOR_BLACK
            def initp(idx, fg, bg=-1):
                try: curses.init_pair(idx, fg, bg)
                except curses.error: curses.init_pair(idx, curses.COLOR_WHITE, -1)
            initp(1, curses.COLOR_BLUE,  -1)
            initp(2, ORANGE,             -1)
            initp(3, curses.COLOR_WHITE, -1)
            initp(4, curses.COLOR_WHITE, -1)
            initp(5, curses.COLOR_WHITE, -1)
            initp(6, GREY,               -1)
        def col_pair(i: int) -> int: return (i + 1) if has_colors else 0

        ansi_pairs = _init_ansi_pairs()

        lists, leaf_info = build_lists()
        def rebuild_and_sync():
            nonlocal lists, leaf_info
            lists, leaf_info = build_lists()
            sync_cursors_to_locks(lists)

        def alert_box(message: str):
            h, w = stdscr.getmaxyx()
            box1_h_total = 3
            y0 = box1_h_total + 2
            safe_w = min(w, w) - 1
            box_w = min(safe_w - 4, max(28, len(message) + 6))
            x0 = max(2, (safe_w - box_w) // 2)
            tl, tr, bl, br = "╭", "╮", "╰", "╯"
            top_line = tl + "─" * (box_w - 2) + tr
            mid_line = "│" + " " * (box_w - 2) + "│"
            bot_line = bl + "─" * (box_w - 2) + br
            stdscr.addnstr(y0, x0, top_line, box_w, curses.A_BOLD)
            stdscr.addnstr(y0 + 1, x0, mid_line, box_w)
            stdscr.addnstr(y0 + 2, x0, mid_line, box_w)
            stdscr.addnstr(y0 + 3, x0, bot_line, box_w, curses.A_BOLD)
            stdscr.addnstr(y0 + 1, x0 + 2, message[: box_w - 4], box_w - 4)
            btn = "[ ok ]"; bx = x0 + (box_w - len(btn)) // 2
            stdscr.addnstr(y0 + 2, bx, btn, len(btn), curses.A_REVERSE | curses.A_BOLD)
            stdscr.refresh()
            while True:
                ch = stdscr.getch()
                if ch in (10, 13, 27): break

        def confirm_box(message: str) -> bool:
            h, w = stdscr.getmaxyx()
            box1_h_total = 3
            y0 = box1_h_total + 2
            safe_w = min(w, w) - 1
            box_w = min(safe_w - 4, max(34, len(message) + 10))
            x0 = max(2, (safe_w - box_w) // 2)
            tl, tr, bl, br = "╭", "╮", "╰", "╯"
            top_line = tl + "─" * (box_w - 2) + tr
            mid_line = "│" + " " * (box_w - 2) + "│"
            bot_line = bl + "─" * (box_w - 2) + br
            stdscr.addnstr(y0, x0, top_line, box_w, curses.A_BOLD)
            stdscr.addnstr(y0 + 1, x0, mid_line, box_w)
            stdscr.addnstr(y0 + 2, x0, mid_line, box_w)
            stdscr.addnstr(y0 + 3, x0, bot_line, box_w, curses.A_BOLD)
            stdscr.addnstr(y0 + 1, x0 + 2, message[: box_w - 4], box_w - 4)
            ok_btn = "[ ok ]"; cancel_btn = "[ cancel ]"
            ok_x = x0 + (box_w - len(ok_btn) - 1 - len(cancel_btn)) // 2
            cancel_x = ok_x + len(ok_btn) + 1
            stdscr.addnstr(y0 + 2, ok_x, ok_btn, len(ok_btn), curses.A_REVERSE | curses.A_BOLD)
            stdscr.addnstr(y0 + 2, cancel_x, cancel_btn, len(cancel_btn), curses.A_DIM | curses.A_BOLD)
            stdscr.refresh()
            while True:
                ch = stdscr.getch()
                if ch in (10, 13): return True
                if ch == 27: return False

        while True:
            eff_cfg = effective_cfg_key(); eff_inst = effective_inst_key(eff_cfg)
            prune_inconsistent_locks(eff_cfg, eff_inst)

            if (eff_cfg != prev_eff_cfg) or (eff_inst != prev_eff_inst):
                locked_by_depth.clear(); counter = 0
                prev_eff_cfg = eff_cfg; prev_eff_inst = eff_inst
                rebuild_and_sync()
                if find_mode:
                    find_query = ""; find_cursor = 0; find_scroll_x = 0
                    rebuild_find_index(eff_cfg, eff_inst); filter_find_results()
                    find_sel = 0; find_list_scroll = 0

            for i in range(N_COLS):
                n = len(lists[i]) if i < len(lists) else 0
                cursor[i] = 0 if n == 0 else max(0, min(cursor[i], n - 1))
                if locked_idx[i] is not None and n:
                    locked_idx[i] = max(0, min(locked_idx[i], n - 1))

            h, w = stdscr.getmaxyx()
            stdscr.erase()
            min_h = 20
            if w < 70 or h < min_h:
                msg = f"Resize terminal (≥ 70x{min_h})."
                stdscr.addnstr(h // 2, max(0, (w - len(msg)) // 2), msg, max(0, w - 1), curses.A_BOLD)
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (ord('q'),): return
                continue

            # Layout
            box1_h_total = 3
            nav_inner_h = 12
            box2_h_total = nav_inner_h + 2
            path_y0, path_y1 = 0, box1_h_total - 1
            nav_y0,  nav_y1  = path_y1 + 1, path_y1 + box2_h_total
            bot_y0,  bot_y1  = nav_y1 + 1, h - 1

            draw_box(stdscr, path_y0, path_y1, w, rounded=True)
            draw_box(stdscr, nav_y0,  nav_y1,  w, rounded=True)
            draw_box(stdscr, bot_y0,  bot_y1,  w, rounded=True)

            inner_w_all = max(1, w - 3)

            input_caret_yx = None
            find_input_yx = None

            # Path / Find-path
            if find_mode:
                cfg = eff_cfg; inst = eff_inst
                draw_find_path(stdscr, path_y0 + 1, 1, inner_w_all, cfg, inst)
            else:
                def draw_path(stdscr, y, x0, w, the_lists: List[List[str]]):
                    stdscr.addnstr(y, x0, " " * max(0, w), max(0, w))
                    parts = compute_effective_path(the_lists)
                    if not parts: return
                    x = x0
                    stdscr.addnstr(y, x, "$", 1, curses.A_BOLD); x += 1
                    p0 = parts[0]; take = min(len(p0), max(0, x0 + w - x))
                    stdscr.addnstr(y, x, p0[:take], take, curses.A_BOLD | curses.color_pair(1)); x += take
                    if x >= x0 + w or len(parts) == 1: return
                    sep = "::"; take = min(len(sep), max(0, x0 + w - x))
                    stdscr.addnstr(y, x, sep[:take], take, curses.A_DIM); x += take
                    if x >= x0 + w: return
                    p1 = parts[1]; take = min(len(p1), max(0, x0 + w - x))
                    stdscr.addnstr(y, x, p1[:take], take, curses.A_BOLD | curses.color_pair(2)); x += take
                    if x >= x0 + w: return
                    for p in parts[2:]:
                        if x >= x0 + w: break
                        sep = "" if (isinstance(p, str) and p.startswith("[")) else "."
                        if sep:
                            stdscr.addnstr(y, x, sep, 1, curses.A_DIM); x += 1
                            if x >= x0 + w: break
                        take = min(len(p), max(0, x0 + w - x))
                        stdscr.addnstr(y, x, p[:take], take, curses.A_BOLD); x += take
                draw_path(stdscr, path_y0 + 1, 1, inner_w_all, lists)

            # ---------- Navigation box ----------
            nav_inner_top = nav_y0 + 1
            nav_inner_bot = nav_y1 - 1
            help_y = nav_inner_top + 11
            top = nav_inner_top + 1
            visible_rows = 10

            base_w = inner_w_all // N_COLS
            rem = inner_w_all % N_COLS
            widths = [(base_w + (1 if i < rem else 0)) for i in range(N_COLS)]
            x_positions = [1]
            for i in range(1, N_COLS): x_positions.append(x_positions[-1] + widths[i - 1])

            vis_lock_idx = [None] * N_COLS
            for col in (2, 3, 4):
                d = left_depth() + (col - 2); tok = locked_by_depth.get(d)
                if tok and lists[col]:
                    try: vis_lock_idx[col] = lists[col].index(tok)
                    except ValueError: vis_lock_idx[col] = None
            vis_lock_idx[0] = locked_idx[0]; vis_lock_idx[1] = locked_idx[1]

            force_idx_col2 = None
            if focus not in (0, 1) and lists[2]:
                force_idx_col2 = max(0, min(cursor[2], len(lists[2])-1))

            if input_mode: help_line = "typing…  enter=submit  esc=cancel"
            elif bool_mode: help_line = "bool: ↑/↓ select • enter=save • esc=cancel"
            elif enum_mode: help_line = "enum: ↑/↓ select • enter=save • esc=cancel"
            elif find_mode: help_line = "FIND: type to filter • ↑/↓ move • →/enter open • esc exit"
            else: help_line = "↑/↓ move • →/enter dive/edit • ← back • f find • esc reset • q quit"
            stdscr.addnstr(help_y, 1 + max(0, (inner_w_all - len(help_line)) // 2),
                           help_line[:inner_w_all], inner_w_all, curses.A_DIM | curses.color_pair(6))

            if find_mode and eff_cfg and eff_inst and find_results and 0 <= find_sel < len(find_results):
                toks, _ = find_results[find_sel]
                pv_lists, pv_cursors, leaf_preview = build_preview_navigator(eff_cfg, eff_inst, toks)

                draw_column(stdscr, x_positions[0], widths[0], top, visible_rows,
                            pv_lists[0], 0, True, pv_cursors[0], None, col_pair(0), suppress_empty_placeholder=False)
                draw_column(stdscr, x_positions[1], widths[1], top, visible_rows,
                            pv_lists[1], 1, True, pv_cursors[1], None, col_pair(1), suppress_empty_placeholder=False)
                draw_column(stdscr, x_positions[2], widths[2], top, visible_rows,
                            pv_lists[2], 2, True, pv_cursors[2], None, col_pair(2), force_focus_idx=force_idx_col2,
                            suppress_empty_placeholder=True)

                if leaf_preview and leaf_preview[0] == 2:
                    val = leaf_preview[1]
                    disp = (str(val).lower() if isinstance(val, bool)
                            else (enum_parts(val)[1] if enum_parts(val) else ("" if val is None else str(val))))
                    draw_value_field(stdscr, x_positions[3], widths[3], top, visible_rows, disp)
                    draw_column(stdscr, x_positions[4], widths[4], top, visible_rows,
                                [], 4, True, 0, None, col_pair(4), suppress_empty_placeholder=True)
                else:
                    draw_column(stdscr, x_positions[3], widths[3], top, visible_rows,
                                pv_lists[3], 3, True, pv_cursors[3], None, col_pair(3),
                                counter_show=(counter if len(pv_lists[3]) > 0 else None),
                                suppress_empty_placeholder=True)
                    if leaf_preview and leaf_preview[0] == 3:
                        val = leaf_preview[1]
                        disp = (str(val).lower() if isinstance(val, bool)
                                else (enum_parts(val)[1] if enum_parts(val) else ("" if val is None else str(val))))
                        draw_value_field(stdscr, x_positions[4], widths[4], top, visible_rows, disp)
                    else:
                        draw_column(stdscr, x_positions[4], widths[4], top, visible_rows,
                                    pv_lists[4], 4, True, pv_cursors[4], None, col_pair(4),
                                    suppress_empty_placeholder=True)

            else:
                draw_column(stdscr, x_positions[0], widths[0], top, visible_rows,
                            lists[0], 0, (focus == 0 and not any([input_mode,bool_mode,enum_mode,find_mode])),
                            cursor[0], vis_lock_idx[0], col_pair(0), suppress_empty_placeholder=False)
                draw_column(stdscr, x_positions[1], widths[1], top, visible_rows,
                            lists[1], 1, (focus == 1 and not any([input_mode,bool_mode,enum_mode,find_mode])),
                            cursor[1], vis_lock_idx[1], col_pair(1), suppress_empty_placeholder=False)
                draw_column(stdscr, x_positions[2], widths[2], top, visible_rows,
                            lists[2], 2, (focus == 2 and not any([input_mode,bool_mode,enum_mode,find_mode])),
                            cursor[2], vis_lock_idx[2], col_pair(2), force_focus_idx=force_idx_col2,
                            suppress_empty_placeholder=True)

                if input_mode:
                    x4 = x_positions[3]; span_w = widths[3] + widths[4]
                    draw_input(stdscr, x4, span_w, top, visible_rows)
                elif bool_mode:
                    x4 = x_positions[3]; span_w = widths[3] + widths[4]
                    cur_idx = 0 if (bool_original_value is True) else 1
                    draw_bool_editor(stdscr, x4, span_w, top, visible_rows, focused=True, current_index=cur_idx)
                elif enum_mode:
                    x4 = x_positions[3]; span_w = widths[3] + widths[4]
                    cur_idx = None
                    if enum_original_value:
                        ep = enum_parts(enum_original_value)
                        if ep:
                            try: cur_idx = enum_variants.index(ep[1])
                            except ValueError: cur_idx = None
                    draw_enum_editor(stdscr, x4, span_w, top, visible_rows, focused=True,
                                     items=enum_variants, type_name=enum_type_name, current_index=cur_idx)
                else:
                    leaf_from_col = None; leaf_val = None
                    if lists[3]:
                        parent3 = parent_node_for_depth(left_depth(), eff_cfg, eff_inst)
                        tok3 = token_at_depth(left_depth()+1)
                        if tok3 and is_container(parent3):
                            ok3, v3 = get_child(parent3, tok3)
                            if ok3 and not is_container(v3): leaf_from_col = 3; leaf_val = v3
                    if leaf_val is None and lists[2]:
                        parent2 = parent_node_for_depth(left_depth()-1, eff_cfg, eff_inst)
                        tok2 = token_at_depth(left_depth())
                        if tok2 and is_container(parent2):
                            ok2, v2 = get_child(parent2, tok2)
                            if ok2 and not is_container(v2): leaf_from_col = 2; leaf_val = v2

                    if leaf_from_col == 2:
                        disp = (str(leaf_val).lower() if isinstance(leaf_val, bool)
                                else (enum_parts(leaf_val)[1] if enum_parts(leaf_val) else ("" if leaf_val is None else str(leaf_val))))
                        draw_value_field(stdscr, x_positions[3], widths[3], top, visible_rows, disp)
                        draw_column(stdscr, x_positions[4], widths[4], top, visible_rows,
                                    lists[4], 4, (focus == 4 and not find_mode), cursor[4], vis_lock_idx[4], col_pair(4),
                                    suppress_empty_placeholder=True)
                    else:
                        counter_title = counter if len(lists[3]) > 0 else None
                        draw_column(stdscr, x_positions[3], widths[3], top, visible_rows,
                                    lists[3], 3, (focus == 3 and not find_mode), cursor[3], vis_lock_idx[3], col_pair(3),
                                    counter_show=counter_title, suppress_empty_placeholder=True)
                        if leaf_from_col == 3 and leaf_val is not None:
                            disp = (str(leaf_val).lower() if isinstance(leaf_val, bool)
                                    else (enum_parts(leaf_val)[1] if enum_parts(leaf_val) else ("" if leaf_val is None else str(leaf_val))))
                            draw_value_field(stdscr, x_positions[4], widths[4], top, visible_rows, disp)
                        else:
                            draw_column(stdscr, x_positions[4], widths[4], top, visible_rows,
                                        lists[4], 4, (focus == 4 and not find_mode), cursor[4], vis_lock_idx[4], col_pair(4),
                                        suppress_empty_placeholder=True)

            # ---------- Bottom box ----------
            bot_inner_top = bot_y0 + 1
            bot_inner_bot = bot_y1 - 1
            bot_inner_h = max(1, bot_inner_bot - bot_inner_top + 1)
            inner_x = 1; inner_w = inner_w_all

            if find_mode:
                res_y0 = bot_inner_top
                res_y1 = bot_inner_bot
                draw_find_results_list(stdscr, res_y0, res_y1, inner_x, inner_w)
            else:
                draw_bottom_menu(stdscr, bot_inner_top, inner_x, inner_w, menu_selected)
                content_y0 = bot_inner_top + 1
                help_y = bot_inner_bot
                content_y1 = help_y - 1

                if menu_selected == MENU_HISTORY:
                    parts = compute_effective_path(lists)
                    hist_error = None
                    if parts and len(parts) >= 2:
                        cfg, inst = parts[0], parts[1]; toks = parts[2:]
                        root_path = build_root_path_for_history(toks)
                        key = (cfg, inst, root_path)
                        if key != hist_last_query_key:
                            ok, out = run_mccs_history(cfg, inst, root_path if toks else "root")
                            if ok:
                                hist_entries = parse_history_output(out)
                                # sort newest first (by numeric version index)
                                hist_entries.sort(key=lambda it: it.idx, reverse=True)
                                hist_sel = 0; hist_scroll = 0; hist_error = None
                                hist_user_forced_show = None  # reset manual override on new query
                            else:
                                hist_entries = []; hist_sel = 0; hist_scroll = 0; hist_error = out
                                hist_user_forced_show = None
                            hist_last_query_key = key
                    else:
                        hist_entries = []; hist_sel = hist_scroll = 0
                        hist_error = "select a module and instance to view history"
                        hist_user_forced_show = None

                    visible_rows_hist = max(0, content_y1 - content_y0 + 1)

                    # Auto-fit changes: if everything fits with changes, show them by default.
                    total_with_changes = sum(1 + len(it.changes) for it in hist_entries)
                    auto_show = total_with_changes <= visible_rows_hist
                    # resolve what to show (user override wins)
                    hist_show_changes = auto_show if (hist_user_forced_show is None) else hist_user_forced_show

                    ensure_sel_visible(visible_rows_hist, hist_show_changes)
                    draw_history_list(stdscr, content_y0, content_y1, inner_x, inner_w, ansi_pairs, hist_show_changes)

                    hl = "w/s move up/down • p pull selected version • k "
                    hl += "hide changes" if hist_show_changes else "show changes"
                    stdscr.addnstr(help_y, inner_x + max(0, (inner_w - len(hl)) // 2), hl[:inner_w], inner_w, curses.A_DIM)

                elif menu_selected == MENU_CHANGES:
                    draw_centered(stdscr, content_y0, content_y1, inner_x, inner_w,
                                  "(active changes) — not implemented yet", curses.A_DIM)
                    stdscr.addnstr(help_y, inner_x, " " * inner_w, inner_w)

                elif menu_selected == MENU_LOGS:
                    draw_centered(stdscr, content_y0, content_y1, inner_x, inner_w,
                                  "(logs) — not implemented yet", curses.A_DIM)
                    stdscr.addnstr(help_y, inner_x, " " * inner_w, inner_w)

            # ---- Final caret placement ----
            if input_mode and input_caret_yx:
                try: curses.curs_set(1); stdscr.move(*input_caret_yx)
                except curses.error: pass
            elif find_mode and find_input_yx:
                try: curses.curs_set(1); stdscr.move(*find_input_yx)
                except curses.error: pass
            else:
                try: curses.curs_set(0)
                except curses.error: pass

            stdscr.refresh()

            # --------- Read one key ----------
            key = stdscr.getch()

            # Global quit / reset (outside editors/find)
            if not (find_mode or input_mode or bool_mode or enum_mode):
                if key in (ord('q'),): return
                if key == 27:  # esc reset path
                    locked_by_depth.clear()
                    locked_idx[0] = locked_idx[1] = None
                    counter = 0
                    focus = 0
                    cursor[:] = [0]*N_COLS
                    rebuild_and_sync()
                    continue

            # Enter FIND
            if not (input_mode or bool_mode or enum_mode or find_mode):
                if key in (ord('f'), ord('F')):
                    if eff_cfg and eff_inst:
                        find_mem_locked_by_depth = locked_by_depth.copy()
                        find_mem_counter = counter
                        find_mem_cursor = cursor.copy()
                        find_mem_focus = focus
                        find_mem_locked_idx = locked_idx.copy()
                        find_mode = True
                        find_query = ""; find_cursor = 0; find_scroll_x = 0; find_input_yx = None
                        rebuild_find_index(eff_cfg, eff_inst); filter_find_results()
                        find_sel = 0; find_list_scroll = 0
                    else:
                        alert_box("select a module and instance first.")
                    continue

            # Bottom menu hotkeys
            if not (input_mode or bool_mode or enum_mode or find_mode):
                if key in (ord('h'), ord('H')): menu_selected = MENU_HISTORY; continue
                elif key in (ord('c'), ord('C')): menu_selected = MENU_CHANGES; continue
                elif key in (ord('l'), ord('L')): menu_selected = MENU_LOGS; continue

            # FIND mode handling
            if find_mode:
                if key == 27:  # esc restore snapshot
                    find_mode = False
                    if find_mem_locked_by_depth is not None:
                        locked_by_depth.clear(); locked_by_depth.update(find_mem_locked_by_depth)
                    if find_mem_counter is not None: counter = find_mem_counter
                    if find_mem_cursor is not None: cursor[:] = find_mem_cursor
                    if find_mem_focus is not None: focus = find_mem_focus
                    if find_mem_locked_idx is not None: locked_idx[:] = find_mem_locked_idx
                    continue
                elif key == curses.KEY_UP:
                    if find_results: find_sel = (find_sel - 1) % len(find_results)
                    continue
                elif key == curses.KEY_DOWN:
                    if find_results: find_sel = (find_sel + 1) % len(find_results)
                    continue
                elif key in (curses.KEY_RIGHT, 10, 13):
                    if find_results and eff_cfg and eff_inst:
                        toks, _ = find_results[find_sel]
                        L = 2 + len(toks)
                        try: locked_idx[0] = lists[0].index(eff_cfg) if lists[0] else 0
                        except ValueError: locked_idx[0] = None
                        try:
                            col1 = list_items(data.get(eff_cfg, {}))
                            locked_idx[1] = col1.index(eff_inst) if col1 else None
                        except ValueError:
                            locked_idx[1] = None
                        locked_by_depth.clear()
                        for i_tok, tok in enumerate(toks):
                            if i_tok < len(toks) - 1:
                                locked_by_depth[3 + i_tok] = tok
                        counter = max(0, L - 4)
                        rebuild_and_sync()
                        if len(toks) >= 2: set_cursor_for_depth_token(L - 1, toks[-2])
                        set_cursor_for_depth_token(L, toks[-1])
                        leaf_col = depth_to_col(L)
                        focus = leaf_col if leaf_col is not None else (depth_to_col(L - 1) or focus)
                        find_mode = False
                    continue
                elif key == curses.KEY_LEFT:
                    if find_cursor > 0: find_cursor -= 1
                elif key == curses.KEY_RIGHT:
                    if find_cursor < len(find_query): find_cursor += 1
                elif key == curses.KEY_HOME: find_cursor = 0
                elif key == curses.KEY_END:  find_cursor = len(find_query)
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    if find_cursor > 0:
                        find_query = find_query[:find_cursor-1] + find_query[find_cursor:]
                        find_cursor -= 1; filter_find_results(); find_sel = 0; find_list_scroll = 0
                elif key == curses.KEY_DC:
                    if find_cursor < len(find_query):
                        find_query = find_query[:find_cursor] + find_query[find_cursor+1:]; filter_find_results()
                elif 32 <= key <= 126:
                    find_query = find_query[:find_cursor] + chr(key) + find_query[find_cursor:]
                    find_cursor += 1; filter_find_results(); find_sel = 0; find_list_scroll = 0
                elif key in (ord('q'),): return
                continue

            # -------------- History tab keys (w/s/k/p only) --------------
            if (not find_mode) and (not any([input_mode, bool_mode, enum_mode])) and (menu_selected == MENU_HISTORY):
                if key == ord('w'):
                    if hist_entries:
                        hist_sel = (hist_sel - 1) % len(hist_entries)
                        content_rows = (bot_y1 - 1) - (bot_y0 + 1) - 1
                        # recompute show flag for visibility calc
                        total_with_changes = sum(1 + len(it.changes) for it in hist_entries)
                        auto_show = total_with_changes <= content_rows
                        show_flag = auto_show if (hist_user_forced_show is None) else hist_user_forced_show
                        ensure_sel_visible(max(0, content_rows), show_flag)
                    continue
                if key == ord('s'):
                    if hist_entries:
                        hist_sel = (hist_sel + 1) % len(hist_entries)
                        content_rows = (bot_y1 - 1) - (bot_y0 + 1) - 1
                        total_with_changes = sum(1 + len(it.changes) for it in hist_entries)
                        auto_show = total_with_changes <= content_rows
                        show_flag = auto_show if (hist_user_forced_show is None) else hist_user_forced_show
                        ensure_sel_visible(max(0, content_rows), show_flag)
                    continue
                if key == ord('k'):
                    # Toggle user preference; start from the currently displayed state
                    content_rows = (bot_y1 - 1) - (bot_y0 + 1) - 1
                    total_with_changes = sum(1 + len(it.changes) for it in hist_entries)
                    auto_show = total_with_changes <= content_rows
                    current_show = auto_show if (hist_user_forced_show is None) else hist_user_forced_show
                    hist_user_forced_show = not current_show
                    ensure_sel_visible(max(0, content_rows), hist_user_forced_show)
                    continue
                if key == ord('p'):
                    if hist_entries and 0 <= hist_sel < len(hist_entries):
                        v = hist_entries[hist_sel].idx
                        parts = compute_effective_path(lists)
                        if parts and len(parts) >= 2:
                            cfg, inst = parts[0], parts[1]
                            if confirm_box(f"discard current changes and pull version {v}?"):
                                ok, msg = run_mccs_pull(cfg, inst, v)
                                if not ok: alert_box(msg or "pull failed")
                                else: hist_last_query_key = None
                        else:
                            alert_box("select a module and instance first.")
                    continue

            # --- Editors ---
            if input_mode:
                if key in (10, 13):
                    if input_target_path is not None:
                        new_val = parse_like(input_original_value, input_text)
                        set_in(data, input_target_path, new_val)
                    if input_saved_cursor is not None: cursor[:] = input_saved_cursor
                    if input_saved_counter is not None: counter = input_saved_counter
                    if input_return_focus is not None: focus = input_return_focus
                    input_mode = False; input_text = ""; input_cursor = input_scroll = 0
                    input_leaf_key = None; input_parent_chain = []
                    input_saved_cursor = None; input_saved_counter = None; input_return_focus = None
                    try: curses.curs_set(0)
                    except curses.error: pass
                    rebuild_and_sync(); continue
                if key == 27:
                    if input_saved_cursor is not None: cursor[:] = input_saved_cursor
                    if input_saved_counter is not None: counter = input_saved_counter
                    if input_return_focus is not None: focus = input_return_focus
                    input_mode = False; input_text = ""; input_cursor = input_scroll = 0
                    input_leaf_key = None; input_parent_chain = []
                    input_saved_cursor = None; input_saved_counter = None; input_return_focus = None
                    try: curses.curs_set(0)
                    except curses.error: pass
                    rebuild_and_sync(); continue
                if key == curses.KEY_LEFT:
                    if input_cursor > 0: input_cursor -= 1
                elif key == curses.KEY_RIGHT:
                    if input_cursor < len(input_text): input_cursor += 1
                elif key == curses.KEY_HOME: input_cursor = 0
                elif key == curses.KEY_END:  input_cursor = len(input_text)
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    if input_cursor > 0:
                        input_text = input_text[:input_cursor-1] + input_text[input_cursor:]; input_cursor -= 1
                elif key == curses.KEY_DC:
                    if input_cursor < len(input_text):
                        input_text = input_text[:input_cursor] + input_text[input_cursor+1:]
                elif 32 <= key <= 126:
                    input_text = input_text[:input_cursor] + chr(key) + input_text[input_cursor:]; input_cursor += 1
                elif key in (ord('q'),): return
                continue

            if bool_mode:
                if key in (curses.KEY_UP, curses.KEY_DOWN): bool_cursor = 1 - bool_cursor
                elif key == 27:
                    if bool_saved_cursor is not None: cursor[:] = bool_saved_cursor
                    if bool_saved_counter is not None: counter = bool_saved_counter
                    if bool_return_focus is not None: focus = bool_return_focus
                    bool_mode = False; bool_target_path = None; bool_original_value = None
                    bool_saved_cursor = None; bool_saved_counter = None; bool_return_focus = None
                    rebuild_and_sync()
                elif key in (10, 13):
                    if bool_target_path is not None:
                        new_val = (bool_cursor == 0); set_in(data, bool_target_path, new_val)
                    if bool_saved_cursor is not None: cursor[:] = bool_saved_cursor
                    if bool_saved_counter is not None: counter = bool_saved_counter
                    if bool_return_focus is not None: focus = bool_return_focus
                    bool_mode = False; bool_target_path = None; bool_original_value = None
                    bool_saved_cursor = None; bool_saved_counter = None; bool_return_focus = None
                    rebuild_and_sync()
                elif key in (ord('q'),): return
                continue

            if enum_mode:
                if key == curses.KEY_UP: enum_cursor = (enum_cursor - 1) % max(1, len(enum_variants))
                elif key == curses.KEY_DOWN: enum_cursor = (enum_cursor + 1) % max(1, len(enum_variants))
                elif key == 27:
                    if enum_saved_cursor is not None: cursor[:] = enum_saved_cursor
                    if enum_saved_counter is not None: counter = enum_saved_counter
                    if enum_return_focus is not None: focus = enum_return_focus
                    enum_mode = False; enum_target_path = None; enum_variants = []; enum_type_name = None
                    enum_original_value = None; enum_saved_cursor = None; enum_saved_counter = None; enum_return_focus = None
                    rebuild_and_sync()
                elif key in (10, 13):
                    if enum_target_path is not None and enum_type_name and enum_variants:
                        chosen = enum_variants[enum_cursor]
                        new_val = f"enum::{enum_type_name}::{chosen}"
                        set_in(data, enum_target_path, new_val)
                    if enum_saved_cursor is not None: cursor[:] = enum_saved_cursor
                    if enum_saved_counter is not None: counter = enum_saved_counter
                    if enum_return_focus is not None: focus = enum_return_focus
                    enum_mode = False; enum_target_path = None; enum_variants = []; enum_type_name = None
                    enum_original_value = None; enum_saved_cursor = None; enum_saved_counter = None; enum_return_focus = None
                    rebuild_and_sync()
                elif key in (ord('q'),): return
                continue

            # --- Navigation (select mode) ---
            if key in (curses.KEY_UP, ord('k')):
                if lists[focus]:
                    cursor[focus] = (cursor[focus] - 1) % len(lists[focus])
                    if focus == 2: cursor[3] = cursor[4] = 0

            elif key in (curses.KEY_DOWN, ord('j')):
                if lists[focus]:
                    cursor[focus] = (cursor[focus] + 1) % len(lists[focus])
                    if focus == 2: cursor[3] = cursor[4] = 0

            elif key in (curses.KEY_LEFT, ord('h')):
                if focus == 3:
                    if counter > 0:
                        prev_idx_tok = None
                        if len(lists[2]) > 0:
                            try: prev_idx_tok = lists[2][cursor[2]]
                            except Exception: prev_idx_tok = None
                        locked_by_depth.pop(left_depth(), None); counter -= 1
                        rebuild_and_sync()
                        if prev_idx_tok is not None: set_cursor_for_depth_token(4, prev_idx_tok)
                    else:
                        locked_by_depth.pop(left_depth(), None); focus = 2; rebuild_and_sync()
                elif focus == 4:
                    if len(lists[4]) > 0:
                        locked_by_depth.pop(5 + counter, None); focus = 3; rebuild_and_sync()
                    else:
                        idx_depth = left_depth(); idx_tok = locked_by_depth.get(idx_depth)
                        if idx_tok is None and len(lists[2]) > 0:
                            try: idx_tok = lists[2][cursor[2]]
                            except Exception: idx_tok = None
                        if idx_tok is not None: locked_by_depth[idx_depth] = idx_tok
                        focus = 3; rebuild_and_sync()
                        if idx_tok is not None: set_cursor_for_depth_token(idx_depth, idx_tok)
                else:
                    if focus > 0:
                        focus -= 1
                        if focus == 1 and len(lists[1]) == 0: focus = 0
                    if focus in (0, 1): locked_idx[focus] = None

            elif key in (curses.KEY_RIGHT, ord('l'), ord('\t'), 10, 13):
                if focus == 0 and lists[0]:
                    if len(lists[1]) > 0: locked_idx[0] = cursor[0]; focus = 1
                elif focus == 1 and lists[1]:
                    locked_idx[1] = cursor[1]; focus = 2
                elif focus == 2 and lists[2]:
                    parent2 = parent_node_for_depth(left_depth()-1, eff_cfg, eff_inst)
                    tok2 = token_at_depth(left_depth())
                    leaf_here = False; val2 = None
                    if tok2 and is_container(parent2):
                        ok2, v2 = get_child(parent2, tok2)
                        if ok2 and not is_container(v2): leaf_here = True; val2 = v2
                    if leaf_here:
                        ep = enum_parts(val2)
                        if isinstance(val2, bool):
                            bool_saved_cursor = cursor.copy(); bool_saved_counter = counter; bool_return_focus = 2
                            full_path = []; eff_cfg2 = effective_cfg_key(); eff_inst2 = effective_inst_key(eff_cfg2)
                            if eff_cfg2: full_path.append(eff_cfg2)
                            if eff_inst2: full_path.append(eff_inst2)
                            full_path.append(tok2); bool_target_path = full_path; bool_original_value = val2
                            bool_cursor = 0 if val2 else 1; bool_mode = True; focus = 4
                        elif ep:
                            enum_saved_cursor = cursor.copy(); enum_saved_counter = counter; enum_return_focus = 2
                            enum_type_name = ep[0]; enum_variants = enum_variants_for(enum_type_name)
                            try: enum_cursor = enum_variants.index(ep[1])
                            except ValueError: enum_cursor = 0
                            full_path = []; eff_cfg2 = effective_cfg_key(); eff_inst2 = effective_inst_key(eff_cfg2)
                            if eff_cfg2: full_path.append(eff_cfg2)
                            if eff_inst2: full_path.append(eff_inst2)
                            full_path.append(tok2); enum_target_path = full_path; enum_original_value = val2
                            enum_mode = True; focus = 4
                        else:
                            input_saved_cursor = cursor.copy(); input_saved_counter = counter; input_return_focus = 2
                            full_path = []; eff_cfg2 = effective_cfg_key(); eff_inst2 = effective_inst_key(eff_cfg2)
                            if eff_cfg2: full_path.append(eff_cfg2)
                            if eff_inst2: full_path.append(eff_inst2)
                            full_path.append(tok2); input_target_path = full_path; input_parent_chain = []
                            input_leaf_key = tok2; input_original_value = val2
                            input_text = "" if val2 is None else str(val2)
                            input_cursor = len(input_text); input_scroll = 0; input_mode = True
                    else:
                        d = left_depth(); locked_by_depth[d] = lists[2][cursor[2]]; focus = 3; rebuild_and_sync()
                elif focus == 3:
                    parent = parent_node_for_depth(left_depth(), eff_cfg, eff_inst)
                    if not is_container(parent) or not lists[3]: pass
                    else:
                        tok_here = lists[3][cursor[3]]; ok, val_here = get_child(parent, tok_here)
                        if not ok: pass
                        elif is_container(val_here):
                            if (is_dict(val_here) and not list_items(val_here)) or (is_list(val_here) and len(val_here) == 0): pass
                            else:
                                d_parent = left_depth()
                                if d_parent not in locked_by_depth and lists[2]:
                                    locked_by_depth[d_parent] = lists[2][cursor[2]]
                                locked_by_depth[left_depth() + 1] = tok_here
                                counter += 1; rebuild_and_sync(); cursor[3] = cursor[4] = 0
                        else:
                            ep = enum_parts(val_here)
                            if isinstance(val_here, bool):
                                bool_saved_cursor = cursor.copy(); bool_saved_counter = counter; bool_return_focus = 3
                                full_path: List[str] = []; eff_cfg2 = effective_cfg_key(); eff_inst2 = effective_inst_key(eff_cfg2)
                                if eff_cfg2: full_path.append(eff_cfg2)
                                if eff_inst2: full_path.append(eff_inst2)
                                node = base_node(eff_cfg2, eff_inst2); dcur = 3; ok2 = True
                                while dcur <= left_depth() and ok2:
                                    if not is_container(node): ok2 = False; break
                                    tok = locked_by_depth.get(dcur)
                                    if not tok: ok2 = False; break
                                    okc, node = get_child(node, tok)
                                    if not okc: ok2 = False; break
                                    full_path.append(tok); dcur += 1
                                if ok2:
                                    full_path.append(tok_here); bool_target_path = full_path
                                    bool_original_value = val_here; bool_cursor = 0 if val_here else 1
                                    bool_mode = True; focus = 4
                            elif ep:
                                enum_saved_cursor = cursor.copy(); enum_saved_counter = counter; enum_return_focus = 3
                                enum_type_name = ep[0]; enum_variants = enum_variants_for(enum_type_name)
                                try: enum_cursor = enum_variants.index(ep[1])
                                except ValueError: enum_cursor = 0
                                full_path: List[str] = []; eff_cfg2 = effective_cfg_key(); eff_inst2 = effective_inst_key(eff_cfg2)
                                if eff_cfg2: full_path.append(eff_cfg2)
                                if eff_inst2: full_path.append(eff_inst2)
                                node = base_node(eff_cfg2, eff_inst2); dcur = 3; ok2 = True
                                while dcur <= left_depth() and ok2:
                                    if not is_container(node): ok2 = False; break
                                    tok = locked_by_depth.get(dcur)
                                    if not tok: ok2 = False; break
                                    okc, node = get_child(node, tok)
                                    if not okc: ok2 = False; break
                                    full_path.append(tok); dcur += 1
                                if ok2:
                                    full_path.append(tok_here); enum_target_path = full_path
                                    enum_original_value = val_here; enum_mode = True; focus = 4
                            else:
                                input_saved_cursor = cursor.copy(); input_saved_counter = counter; input_return_focus = 3
                                full_path: List[str] = []; eff_cfg2 = effective_cfg_key(); eff_inst2 = effective_inst_key(eff_cfg2)
                                if eff_cfg2: full_path.append(eff_cfg2)
                                if eff_inst2: full_path.append(eff_inst2)
                                node = base_node(eff_cfg2, eff_inst2); parent_chain: List[str] = []; ok2 = True; dcur = 3
                                while dcur <= left_depth() and ok2:
                                    if not is_container(node): ok2 = False; break
                                    tok = locked_by_depth.get(dcur)
                                    if not tok: ok2 = False; break
                                    okc, node = get_child(node, tok)
                                    if not okc: ok2 = False; break
                                    full_path.append(tok); parent_chain.append(tok); dcur += 1
                                if ok2:
                                    full_path.append(tok_here); input_target_path = full_path; input_parent_chain = parent_chain
                                    input_leaf_key = tok_here; input_original_value = val_here
                                    input_text = "" if val_here is None else str(val_here)
                                    input_cursor = len(input_text); input_scroll = 0; input_mode = True

            lists, leaf_info = build_lists()

    curses.wrapper(main)
    return data

if __name__ == "__main__":
    # Keep these module-level so they stay alive for later features
    DATA_CCFS = {}
    VIEW_CCFS = {}

    try:
        # --- Load from your MCCS interface ---
        from mccs_interface_python2 import Host as mccs
        host = mccs()

        def cache_ccf(configuration, instance, ccf, data_ccfs, view_ccfs):
            if configuration not in data_ccfs:
                data_ccfs[configuration] = {}
                view_ccfs[configuration] = {}
            data_ccfs[configuration][instance] = ccf
            view_ccfs[configuration][instance] = ccf.content  # dict-like content used by the TUI

        def load_ccfs(data_ccfs, view_ccfs):
            configurations = host.get_configurations()
            for cfg in configurations:
                instances = host.get_instances(cfg)
                for instance in instances:
                    ccf = host.pull(cfg, instance)
                    cache_ccf(cfg, instance, ccf, data_ccfs, view_ccfs)

        def load_root():
            data_ccfs = {}
            view_ccfs = {}
            load_ccfs(data_ccfs, view_ccfs)
            return data_ccfs, view_ccfs

        DATA_CCFS, VIEW_CCFS = load_root()
        run_json_tui(VIEW_CCFS)

    except Exception as e:
        # Start TUI with empty data and show a centered alert in the navigator box
        err = f"Failed to load MCCS data:\n{e}\n\nPress enter or esc to continue."
        run_json_tui({}, startup_alert=err)
