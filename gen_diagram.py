import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── palette ──────────────────────────────────────────────
BG       = '#0d1117'
GREEN    = '#3fb950'
GREEN_D  = '#0f2820'
GREEN_B  = '#238636'
BLUE     = '#58a6ff'
BLUE_D   = '#0d1f3c'
BLUE_B   = '#1f6feb'
PURPLE   = '#bc8cff'
PURPLE_D = '#130d2a'
PURPLE_B = '#6e40c9'
YELLOW   = '#e3b341'
YELLOW_D = '#1c1600'
YELLOW_B = '#b08800'
RED      = '#f85149'
RED_D    = '#1c0505'
RED_B    = '#da3633'
GRAY     = '#8b949e'
WHITE    = '#e6edf3'

fig = plt.figure(figsize=(18, 26), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18)
ax.set_ylim(0, 26)
ax.axis('off')
ax.set_facecolor(BG)

def box(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.25):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)

def label(ax, x, y, text, color=WHITE, size=9, ha='left', va='center',
          bold=False, mono=False):
    family = 'monospace' if mono else 'DejaVu Sans'
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va,
            fontfamily=family, fontweight=weight, zorder=5)

def section_label(ax, x, y, text):
    label(ax, x, y, text, color=GRAY, size=8, bold=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5), zorder=4)

# ══════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════
label(ax, 9, 25.4, 'CUDA Kernel Hierarchy', color=GREEN,
      size=18, ha='center', bold=True)
label(ax, 9, 25.0,
      'whoami<<<dim3(2,3,4), dim3(4,4,4)>>>()   —   24 blocks · 64 threads/block · 1,536 total threads',
      color=GRAY, size=9, ha='center', mono=True)

# ══════════════════════════════════════════════════════════
#  ① GRID
# ══════════════════════════════════════════════════════════
section_label(ax, 0.4, 24.4, '①  SOFTWARE — GRID')
box(ax, 0.3, 20.8, 17.4, 3.4, GREEN_D, GREEN_B, lw=2, radius=0.3)
label(ax, 0.8, 24.0, 'GRID', color=GREEN, size=11, bold=True)
label(ax, 0.8, 23.65,
      'gridDim = (2, 3, 4)  →  24 blocks distributed across all SMs',
      color=GRAY, size=8, mono=True)

# draw block grid (4 z-slices × 2×3 grid)
z_labels = ['z = 0', 'z = 1', 'z = 2', 'z = 3']
col_w, row_h = 1.85, 0.55
start_x = 0.75
start_y = 22.85
for zi, zl in enumerate(z_labels):
    ox = start_x + zi * (2 * col_w + 0.45)
    label(ax, ox + col_w - 0.1, start_y + 0.25, zl, color=GRAY, size=7.5,
          ha='center', mono=True)
    for yi in range(3):
        for xi in range(2):
            bx = ox + xi * col_w
            by = start_y - 0.1 - yi * row_h - row_h
            box(ax, bx, by, col_w - 0.1, row_h - 0.08, '#1a3a28', GREEN_B,
                lw=1, radius=0.12)
            label(ax, bx + (col_w - 0.1) / 2, by + (row_h - 0.08) / 2,
                  f'({xi},{yi},{zi})', color=GREEN, size=7, ha='center', mono=True)

# ══════════════════════════════════════════════════════════
#  arrow 1
# ══════════════════════════════════════════════════════════
arrow(ax, 9, 20.8, 9, 20.35)
label(ax, 9, 20.18,
      'GPU scheduler distributes blocks to SMs',
      color=GRAY, size=8, ha='center')

# ══════════════════════════════════════════════════════════
#  ② SMs
# ══════════════════════════════════════════════════════════
section_label(ax, 0.4, 19.85, '②  HARDWARE — STREAMING MULTIPROCESSORS  (RTX 3060: 28 SMs, 128 cores each)')

sm_configs = [
    ('SM 0', 'Block (0,0,0)', 0.3),
    ('SM 1', 'Block (1,0,0)', 6.35),
]

for sm_title, blk_title, sx in sm_configs:
    box(ax, sx, 15.5, 5.7, 4.1, BLUE_D, BLUE_B, lw=2, radius=0.3)
    label(ax, sx + 0.25, 19.4, sm_title, color=BLUE, size=10, bold=True)
    label(ax, sx + 0.25, 19.1, '128 CUDA Cores · 48 KB Shared · 65536 regs',
          color=GRAY, size=7.5, mono=True)

    # block inside SM
    box(ax, sx + 0.2, 16.55, 5.3, 2.35, '#112240', BLUE_B, lw=1.2, radius=0.2)
    label(ax, sx + 0.45, 18.75, blk_title + '  —  64 threads = 2 warps',
          color='#79c0ff', size=8, mono=True)

    # warp rows
    for wi, (wlabel, alpha) in enumerate([('Warp 0', 1.0), ('Warp 1', 0.4)]):
        wy = 17.9 - wi * 0.95
        label(ax, sx + 0.45, wy, wlabel, color=GRAY, size=7.5, mono=True)
        for ti in range(32):
            tx = sx + 1.15 + ti * 0.135
            fc = BLUE if alpha == 1.0 else '#1a2a3a'
            ec = BLUE_B if alpha == 1.0 else '#1a2a3a'
            box(ax, tx, wy - 0.22, 0.11, 0.35, fc, ec, lw=0.5, radius=0.04)

    # queued block
    box(ax, sx + 0.2, 15.6, 5.3, 0.75, BG, BLUE_B, lw=1, radius=0.15)
    label(ax, sx + 2.85, 15.97,
          'Next block — queued ···',
          color='#4a7cbf', size=8, ha='center', mono=True)

    # resource chips
    chips = ['128 CUDA Cores', '48 KB Shared', '65536 Regs', 'max 1536 threads']
    for ci, chip in enumerate(chips):
        cx = sx + 0.28 + ci * 1.37
        box(ax, cx, 15.15, 1.28, 0.28, '#0d2a4a', BLUE_B, lw=0.8, radius=0.12)
        label(ax, cx + 0.64, 15.29, chip, color=BLUE, size=6.5,
              ha='center', mono=True)

# faded SM rest
box(ax, 12.4, 15.5, 5.3, 4.1, BLUE_D, BLUE_B, lw=1.5, radius=0.3)
ax.add_patch(mpatches.FancyBboxPatch((12.4, 15.5), 5.3, 4.1,
             boxstyle='round,pad=0,rounding_size=0.3',
             facecolor=BG, edgecolor=BG, linewidth=0, alpha=0.5, zorder=3))
label(ax, 15.05, 18.5, 'SM 2 … SM 27', color='#2a4a6a', size=10,
      bold=True, ha='center')
label(ax, 15.05, 18.0, 'remaining blocks distributed here',
      color='#2a4060', size=8, ha='center', mono=True)
for i, t in enumerate(['Block (0,2,0) ···', 'Block (1,2,0) ···',
                        'Block (0,0,1) ···', '···']):
    box(ax, 12.65, 17.4 - i * 0.48, 4.8, 0.38, BG, '#1a3a5a', lw=0.8, radius=0.1)
    label(ax, 15.05, 17.59 - i * 0.48, t, color='#2a4a6a',
          size=7.5, ha='center', mono=True)

# ══════════════════════════════════════════════════════════
#  arrow 2
# ══════════════════════════════════════════════════════════
arrow(ax, 9, 15.5, 9, 15.05)
label(ax, 9, 14.88,
      'each block splits into warps of 32 threads',
      color=GRAY, size=8, ha='center')

# ══════════════════════════════════════════════════════════
#  ③ WARP DETAIL
# ══════════════════════════════════════════════════════════
section_label(ax, 0.4, 14.55, '③  HARDWARE — WARP EXECUTION (SIMT)')
box(ax, 0.3, 10.8, 17.4, 3.55, PURPLE_D, PURPLE_B, lw=2, radius=0.3)
label(ax, 0.8, 14.2, 'Warp Scheduler — Latency Hiding', color=PURPLE,
      size=10, bold=True)
label(ax, 0.8, 13.88,
      '32 threads fire the SAME instruction on 32 cores each cycle. SM switches warps to hide memory latency.',
      color=GRAY, size=8, mono=True)

cycle_data = [
    ('Cycle 1', True,  'Warp 0 executing  '),
    ('Cycle 2', False, 'Warp 0 waits (mem) → Warp 1 runs'),
    ('Cycle N', True,  'Warp 0 resumes    '),
]
for ri, (clabel, active, note) in enumerate(cycle_data):
    ry = 13.3 - ri * 0.83
    label(ax, 0.55, ry, clabel, color=GRAY, size=8, mono=True)
    for ti in range(32):
        tx = 1.55 + ti * 0.36
        fc = PURPLE_B if active else '#1a1229'
        ec = PURPLE   if active else '#2a1f45'
        box(ax, tx, ry - 0.27, 0.3, 0.46, fc, ec, lw=0.7, radius=0.06)
        label(ax, tx + 0.15, ry - 0.04, f't{ti}', color=WHITE if active else '#4a3570',
              size=5.5, ha='center', mono=True)
    label(ax, 13.25, ry, note, color=PURPLE if active else '#4a7cbf',
          size=8, mono=True)

# ══════════════════════════════════════════════════════════
#  ④ MEMORY
# ══════════════════════════════════════════════════════════
section_label(ax, 0.4, 10.5, '④  MEMORY HIERARCHY — speed vs scope')
box(ax, 0.3, 7.1, 17.4, 3.2, YELLOW_D, YELLOW_B, lw=2, radius=0.3)
label(ax, 0.8, 10.15, 'Memory Levels', color=YELLOW, size=10, bold=True)

mem_rows = [
    ('Thread',   'Registers',       '~1 cycle · private/thread · ~42 regs/thread',
     YELLOW, '#2a1a00', YELLOW_B, 1.00, 'fastest'),
    ('Block',    'Shared Memory',   '~5 cycles · 48 KB/block · use __shared__',
     GREEN,  '#1a2a0a', GREEN_B,  0.82, 'fast'),
    ('All SMs',  'L2 Cache',        '~80 cycles · 2.25 MB · automatic',
     BLUE,   BLUE_D,   BLUE_B,   0.62, 'medium'),
    ('Grid',     'Global Memory',   '~300 cycles · 11.9 GB VRAM',
     RED,    RED_D,    RED_B,    0.44, 'slow'),
    ('CPU<->GPU','PCIe / cudaMemcpy','us latency · host RAM',
     GRAY,   '#1a1a1a', '#3a3a3a', 0.28, 'slowest'),
]
for mi, (scope, name, detail, col, fc, ec, w_frac, speed) in enumerate(mem_rows):
    my = 9.6 - mi * 0.54
    label(ax, 0.55, my, scope, color=GRAY, size=7.5, mono=True, ha='left')
    bar_w = 11.5 * w_frac
    box(ax, 1.7, my - 0.22, bar_w, 0.4, fc, ec, lw=1.2, radius=0.1)
    label(ax, 1.85, my - 0.02,
          f'{name}  —  {detail}', color=col, size=8, mono=True)
    label(ax, 16.9, my - 0.02, speed, color=col, size=8, ha='right')

# ══════════════════════════════════════════════════════════
#  ⑤ FORMULA
# ══════════════════════════════════════════════════════════
section_label(ax, 0.4, 6.8, '⑤  GLOBAL THREAD ID FORMULA  (from whoami.cu)')
box(ax, 0.3, 3.5, 17.4, 3.1, RED_D, RED_B, lw=2, radius=0.3)
label(ax, 0.8, 6.45, 'Unique ID per thread — 3D grid + 3D block',
      color=RED, size=10, bold=True)

lines = [
    ('block_id',     '= blockIdx.x  +  blockIdx.y × gridDim.x  +  blockIdx.z × gridDim.x × gridDim.y'),
    ('block_offset', '= block_id  ×  (blockDim.x × blockDim.y × blockDim.z)'),
    ('thread_offset','= threadIdx.x  +  threadIdx.y × blockDim.x  +  threadIdx.z × blockDim.x × blockDim.y'),
    ('global_id',    '= block_offset  +  thread_offset'),
]
for li, (lhs, rhs) in enumerate(lines):
    ly = 5.9 - li * 0.6
    box(ax, 0.5, ly - 0.24, 17.1, 0.46, '#2a0a0a', RED_B, lw=0.8, radius=0.1)
    label(ax, 0.75, ly - 0.01, lhs, color='#ff7b72', size=9, mono=True)
    label(ax, 3.3,  ly - 0.01, rhs, color='#ffa198', size=9, mono=True)

# ══════════════════════════════════════════════════════════
#  LEGEND
# ══════════════════════════════════════════════════════════
legend_items = [
    (GREEN,  'Grid / Block'),
    (BLUE,   'SM / Hardware'),
    (PURPLE, 'Warp / SIMT'),
    (YELLOW, 'Memory'),
    (RED,    'Thread ID Formula'),
]
lx = 1.5
for col, text in legend_items:
    box(ax, lx, 2.9, 0.35, 0.35, col, col, lw=0, radius=0.06)
    label(ax, lx + 0.5, 3.07, text, color=GRAY, size=8)
    lx += 3.1

plt.savefig('/root/cuda-101/kernel_hierarchy.png',
            dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print('saved kernel_hierarchy.png')
