"""
Convert Markdown (.md) files from Restored_output into PNG images using
Pillow only. Renders headers, tables, bold/italic text, and paragraphs
in a clean government-document style.

Output: D:\PDF Restoration\Restored_output\output image\<name>.png
"""
import os
import re
import glob
import textwrap
from PIL import Image, ImageDraw, ImageFont

# ─── Paths ────────────────────────────────────────────────────────────────────
MD_DIR  = r"D:\PDF Restoration\Restored_output"
OUT_DIR = r"D:\PDF Restoration\Restored_output\output image"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Layout constants ────────────────────────────────────────────────────────
PAGE_W       = 2480        # ~8.27 inch at 300 dpi (A4 width)
MARGIN_LEFT  = 180
MARGIN_RIGHT = 180
MARGIN_TOP   = 160
MARGIN_BOT   = 160
CONTENT_W    = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT
DPI          = 300

BG_COLOR     = (255, 255, 255)
TEXT_COLOR   = (20, 20, 20)
TABLE_BORDER = (80, 80, 80)
TABLE_HEADER_BG = (225, 225, 225)
LINE_COLOR   = (180, 180, 180)

# ─── Fonts (use system fonts available on Windows) ───────────────────────────
def _load_font(name, size):
    """Try to load a TrueType font by name from common Windows font paths."""
    paths = [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name),
        name,
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()

FONT_BODY      = _load_font("times.ttf", 42)
FONT_BODY_BOLD = _load_font("timesbd.ttf", 42)
FONT_BODY_IT   = _load_font("timesi.ttf", 42)
FONT_BODY_BI   = _load_font("timesbi.ttf", 42)
FONT_H1        = _load_font("timesbd.ttf", 56)
FONT_H2        = _load_font("timesbd.ttf", 50)
FONT_H3        = _load_font("timesbd.ttf", 46)
FONT_TABLE     = _load_font("times.ttf", 36)
FONT_TABLE_H   = _load_font("timesbd.ttf", 36)


# ─── Markdown parser ─────────────────────────────────────────────────────────
def parse_md(text: str):
    """Parse markdown text into a list of block elements."""
    lines = text.replace('\r\n', '\n').split('\n')
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Empty line
        if not stripped:
            i += 1
            continue

        # Table block: collect consecutive | ... | lines
        if stripped.startswith('|') and stripped.endswith('|'):
            table_rows = []
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith('|') and s.endswith('|'):
                    cells = [c.strip() for c in s.strip('|').split('|')]
                    # Skip separator rows
                    if all(re.match(r'^:?-+:?$', c) for c in cells):
                        i += 1
                        continue
                    table_rows.append(cells)
                    i += 1
                else:
                    break
            if table_rows:
                blocks.append(('table', table_rows))
            continue

        # Header
        m = re.match(r'^(#{1,6})\s+(.*)', stripped)
        if m:
            level = len(m.group(1))
            blocks.append(('header', level, m.group(2)))
            i += 1
            continue

        # Paragraph: accumulate until empty line or structural element
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            s = lines[i].strip()
            if not s or s.startswith('#') or (s.startswith('|') and s.endswith('|')):
                break
            para_lines.append(s)
            i += 1
        blocks.append(('paragraph', ' '.join(para_lines)))

    return blocks


# ─── Rich-text drawing utilities ─────────────────────────────────────────────
def _split_inline(text):
    """Split text into segments: (text, bold, italic)."""
    segments = []
    # Pattern: **bold**, *italic*, or plain text
    pattern = re.compile(r'(\*\*(.+?)\*\*|\*(.+?)\*|(\^(.+?)\^))')
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            segments.append((text[last:m.start()], False, False))
        if m.group(2) is not None:  # bold
            segments.append((m.group(2), True, False))
        elif m.group(3) is not None:  # italic
            segments.append((m.group(3), False, True))
        elif m.group(5) is not None:  # superscript – treat as plain
            segments.append((m.group(5), False, False))
        last = m.end()
    if last < len(text):
        segments.append((text[last:], False, False))
    return segments


def _get_font(bold, italic, is_table=False):
    if is_table:
        return FONT_TABLE_H if bold else FONT_TABLE
    if bold and italic:
        return FONT_BODY_BI
    if bold:
        return FONT_BODY_BOLD
    if italic:
        return FONT_BODY_IT
    return FONT_BODY


def _text_width(text, font):
    """Get text width using the font."""
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def _line_height(font):
    bbox = font.getbbox("Ag")
    return int((bbox[3] - bbox[1]) * 1.45)


def _draw_rich_text(draw, x, y, text, max_width, base_font=None, is_table=False):
    """Draw inline-formatted text with word wrapping. Returns the height used."""
    if base_font is None:
        base_font = FONT_TABLE if is_table else FONT_BODY
    lh = _line_height(base_font)
    segments = _split_inline(text)

    # Flatten segments into words with formatting
    words = []
    for seg_text, bold, italic in segments:
        for word in seg_text.split():
            words.append((word, bold, italic))

    cur_x = x
    cur_y = y
    space_w = _text_width(' ', base_font)

    for word, bold, italic in words:
        font = _get_font(bold, italic, is_table)
        w = _text_width(word, font)

        if cur_x + w > x + max_width and cur_x > x:
            cur_x = x
            cur_y += lh

        draw.text((cur_x, cur_y), word, fill=TEXT_COLOR, font=font)
        cur_x += w + space_w

    return cur_y + lh - y


# ─── Table renderer ──────────────────────────────────────────────────────────
def _draw_table(draw, x, y, rows, content_w):
    """Draw a table. First row is header. Returns total height."""
    if not rows:
        return 0

    num_cols = max(len(r) for r in rows)
    # Pad rows to num_cols
    rows = [r + [''] * (num_cols - len(r)) for r in rows]

    cell_pad = 16
    border_w = 2
    lh = _line_height(FONT_TABLE)

    # Calculate column widths proportionally
    col_widths = []
    total_weight = 0
    for c in range(num_cols):
        max_len = max(len(rows[r][c]) for r in range(len(rows)))
        weight = max(max_len, 4)
        col_widths.append(weight)
        total_weight += weight

    # Convert weights to pixel widths
    usable_w = content_w - (num_cols + 1) * border_w
    col_widths = [max(int(w / total_weight * usable_w), 80) for w in col_widths]
    # Adjust last column to fill remaining space
    col_widths[-1] = usable_w - sum(col_widths[:-1])

    # First pass: compute row heights
    row_heights = []
    for ri, row in enumerate(rows):
        max_h = lh
        for ci, cell in enumerate(row):
            cell_w = col_widths[ci] - 2 * cell_pad
            # Estimate wrapped height
            font = FONT_TABLE_H if ri == 0 else FONT_TABLE
            words = cell.split()
            cur_w = 0
            lines = 1
            space_w = _text_width(' ', font)
            for word in words:
                ww = _text_width(word, font)
                if cur_w + ww > cell_w and cur_w > 0:
                    lines += 1
                    cur_w = ww + space_w
                else:
                    cur_w += ww + space_w
            h = lines * lh + 2 * cell_pad
            max_h = max(max_h, h)
        row_heights.append(max_h)

    total_h = sum(row_heights) + (len(rows) + 1) * border_w

    # Draw table
    cy = y
    for ri, row in enumerate(rows):
        cx = x
        rh = row_heights[ri]

        # Header background
        if ri == 0:
            row_x = x
            for cw in col_widths:
                draw.rectangle([row_x, cy, row_x + cw + border_w, cy + rh], fill=TABLE_HEADER_BG)
                row_x += cw + border_w

        # Draw cells
        for ci, cell in enumerate(row):
            cw = col_widths[ci]

            # Cell border
            draw.rectangle([cx, cy, cx + cw, cy + rh], outline=TABLE_BORDER, width=border_w)

            # Cell text
            font = FONT_TABLE_H if ri == 0 else FONT_TABLE
            clean = re.sub(r'\*\*(.+?)\*\*', r'\1', cell)
            clean = re.sub(r'\*(.+?)\*', r'\1', clean)
            _draw_rich_text(draw, cx + cell_pad, cy + cell_pad,
                            cell, cw - 2 * cell_pad, font, is_table=True)

            cx += cw + border_w

        cy += rh + border_w

    return total_h


# ─── Main rendering function ─────────────────────────────────────────────────
def render_md_to_image(md_path: str, out_path: str):
    """Render a markdown file to a PNG image."""
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    blocks = parse_md(md_text)

    # Two-pass rendering: first pass computes total height, second pass draws
    def _render_pass(draw, is_measure=False):
        y = MARGIN_TOP

        for block in blocks:
            btype = block[0]

            if btype == 'header':
                _, level, text = block
                font = {1: FONT_H1, 2: FONT_H2, 3: FONT_H3}.get(level, FONT_H3)
                y += 12  # spacing before header
                if not is_measure:
                    # Strip inline markers for header
                    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
                    clean = re.sub(r'\*(.+?)\*', r'\1', clean)
                    draw.text((MARGIN_LEFT, y), clean, fill=TEXT_COLOR, font=font)
                y += _line_height(font) + 8

            elif btype == 'table':
                _, rows = block
                if is_measure:
                    # Estimate height
                    h = _draw_table(draw, MARGIN_LEFT, y, rows, CONTENT_W)
                else:
                    h = _draw_table(draw, MARGIN_LEFT, y, rows, CONTENT_W)
                y += h + 16

            elif btype == 'paragraph':
                _, text = block
                if not is_measure:
                    h = _draw_rich_text(draw, MARGIN_LEFT, y, text, CONTENT_W)
                else:
                    # Estimate paragraph height
                    h = _draw_rich_text(draw, MARGIN_LEFT, y, text, CONTENT_W)
                y += h + 14

        return y + MARGIN_BOT

    # Measure pass
    measure_img = Image.new('RGB', (PAGE_W, 10000), BG_COLOR)
    measure_draw = ImageDraw.Draw(measure_img)
    total_h = _render_pass(measure_draw, is_measure=True)
    measure_img.close()

    # Actual render
    page_h = max(total_h, 800)
    img = Image.new('RGB', (PAGE_W, page_h), BG_COLOR)
    draw = ImageDraw.Draw(img)
    _render_pass(draw, is_measure=False)

    img.save(out_path, 'PNG', dpi=(DPI, DPI))
    print(f"  Saved: {out_path}  ({img.width}x{img.height})")
    img.close()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    md_files = sorted(glob.glob(os.path.join(MD_DIR, "*.md")))
    if not md_files:
        print("No .md files found in", MD_DIR)
        exit(1)

    print(f"Converting {len(md_files)} Markdown files to PNG images...\n")
    for md_path in md_files:
        basename = os.path.splitext(os.path.basename(md_path))[0]
        out_path = os.path.join(OUT_DIR, f"{basename}.png")
        print(f"[+] {os.path.basename(md_path)}")
        render_md_to_image(md_path, out_path)

    print(f"\nAll images saved to: {OUT_DIR}")
