"""Local desktop UI with a transparent, click-through game overlay."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

from .engine import Engine, FrameState
from .windows import make_overlay_clickthrough

BG = '#0d111a'
SURFACE = '#171e2b'
TEXT = '#eff4ff'
MUTED = '#9cabbd'
ACCENT = '#65b8ff'
GREEN = '#77dfa8'
RED = '#ff8f9e'
TRANSPARENT = '#010203'


class Overlay:
    def __init__(self, root: tk.Tk) -> None:
        self.top = tk.Toplevel(root)
        self.top.withdraw()
        self.top.overrideredirect(True)
        self.top.configure(bg=TRANSPARENT)
        self.top.attributes('-topmost', True)
        self.top.wm_attributes('-transparentcolor', TRANSPARENT)
        self.canvas = tk.Canvas(self.top, bg=TRANSPARENT, highlightthickness=0, bd=0)
        self.canvas.pack(fill='both', expand=True)
        self._shown = False
        self._last_geometry: str | None = None
        # Prepare click-through and no-activate styles before first display.
        self.top.update_idletasks()
        make_overlay_clickthrough(self.top.winfo_id())

    def draw(self, snap: FrameState, enabled: bool) -> None:
        if not enabled or snap.state not in ('PREVIEW', 'ACTIVE') or not snap.region[2]:
            if self._shown:
                self.top.withdraw()
                self._shown = False
            return
        x, y, w, h = snap.region
        geometry = f'{w}x{h}{x:+d}{y:+d}'
        if geometry != self._last_geometry:
            self.top.geometry(geometry)
            self.canvas.configure(width=w, height=h)
            self._last_geometry = geometry
        # Repeated Tk deiconify() calls on an always-on-top Toplevel may
        # disturb game focus. Only show the overlay on a hidden->shown transition.
        if not self._shown:
            self.top.deiconify()
            self.top.update_idletasks()
            make_overlay_clickthrough(self.top.winfo_id())
            self._shown = True
        self.canvas.delete('all')
        cx, cy = w / 2, h / 2
        self.canvas.create_line(cx - 8, cy, cx + 8, cy, fill=ACCENT, width=1)
        self.canvas.create_line(cx, cy - 8, cx, cy + 8, fill=ACCENT, width=1)
        for box in snap.boxes:
            self.canvas.create_rectangle(box.x1, box.y1, box.x2, box.y2,
                                         outline=GREEN, width=2)
            self.canvas.create_text(box.x1 + 2, max(11, box.y1 - 9),
                                    text=f'{box.score:.0%}', fill=GREEN,
                                    font=('Segoe UI', 9, 'bold'), anchor='w')
        if snap.target:
            px, py = snap.target
            self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4,
                                    outline=RED if snap.control_active else ACCENT, width=2)
        self.canvas.create_text(10, 12, anchor='w', fill=GREEN if snap.control_active else ACCENT,
                                font=('Segoe UI', 10, 'bold'),
                                text=f"SCREEN SIGHT  |  {snap.fps:.0f} FPS  |  {len(snap.boxes)} people  |  {'MOUSE ON' if snap.control_active else 'VIEW ONLY'}")

    def close(self) -> None:
        self.top.destroy()


class MainWindow:
    def __init__(self) -> None:
        self.engine = Engine()
        self.root = tk.Tk()
        self.root.title('ScreenSight 0.1 · Local Vision Lab')
        self.root.geometry('502x760')
        self.root.resizable(False, False)
        self.root.configure(bg=BG)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.overlay = Overlay(self.root)
        self._last_state = ''
        self.scrollbar = tk.Scrollbar(self.root, orient='vertical', bg='#344153',
                                     activebackground=ACCENT, troughcolor=BG)
        self.scrollbar.pack(side='right', fill='y')
        self.scroller = tk.Canvas(self.root, bg=BG, highlightthickness=0, bd=0,
                                  yscrollcommand=self.scrollbar.set)
        self.scroller.pack(side='left', fill='both', expand=True)
        self.scrollbar.configure(command=self.scroller.yview)
        self.content = tk.Frame(self.scroller, bg=BG)
        self.content_window = self.scroller.create_window((0, 0), window=self.content, anchor='nw')
        self.content.bind('<Configure>', lambda _ev: self.scroller.configure(scrollregion=self.scroller.bbox('all')))
        self.scroller.bind('<Configure>', lambda ev: self.scroller.itemconfigure(self.content_window, width=ev.width))
        self.scroller.bind_all('<MouseWheel>', lambda ev: self.scroller.yview_scroll(-int(ev.delta / 120), 'units'))
        self.enabled = tk.BooleanVar(value=False)
        self.preview = tk.BooleanVar(value=True)
        self.show_overlay = tk.BooleanVar(value=True)
        self.window_title = tk.StringVar(value='Call of Duty|Black Ops 7')
        self.model_path = tk.StringVar(value=self.engine.options().model)
        self.confidence = tk.DoubleVar(value=.46)
        self.strength = tk.DoubleVar(value=.22)
        self.aim_height = tk.DoubleVar(value=.21)
        self.roi = tk.IntVar(value=480)
        self.max_step = tk.IntVar(value=23)
        self.status = tk.StringVar(value='STOPPED')
        self.details = tk.StringVar(value='Start vision to load a local ONNX model.')
        self.stats = tk.StringVar(value='GPU/CPU: —    FPS: —    Detected: —')
        self._build()
        self.root.after(65, self._poll)

    def _label(self, parent: tk.Misc, text: str, font_size: int = 10, color: str = MUTED, **kw: object) -> tk.Label:
        return tk.Label(parent, text=text, bg=parent.cget('bg'), fg=color,
                        font=('Segoe UI', font_size), **kw)

    def _card(self, title: str) -> tk.Frame:
        outer = tk.Frame(self.content, bg=SURFACE, padx=16, pady=12)
        outer.pack(fill='x', padx=18, pady=(0, 12))
        self._label(outer, title, 11, TEXT, anchor='w').pack(fill='x', pady=(0, 7))
        return outer

    def _slider(self, card: tk.Frame, label: str, var: tk.Variable,
                from_: float, to: float, resolution: float) -> None:
        r = tk.Frame(card, bg=SURFACE)
        r.pack(fill='x', pady=2)
        self._label(r, label, 9, MUTED, anchor='w').pack(side='left')
        value = self._label(r, '', 9, TEXT)
        value.pack(side='right')
        def change(v: str) -> None:
            n = float(v)
            value.config(text=f'{n:.2f}' if resolution < 1 else str(int(n)))
            self._configure()
        bar = tk.Scale(card, orient='horizontal', from_=from_, to=to, resolution=resolution,
                       showvalue=0, variable=var, command=change, bg=SURFACE, fg=TEXT,
                       activebackground=ACCENT, troughcolor='#344153', highlightthickness=0,
                       bd=0, sliderlength=18, relief='flat', length=455)
        bar.pack(fill='x')
        change(str(var.get()))

    def _build(self) -> None:
        head = tk.Frame(self.content, bg=BG, padx=20, pady=16)
        head.pack(fill='x')
        self._label(head, 'SCREEN SIGHT', 23, TEXT, anchor='w').pack(fill='x')
        self._label(head, 'Offline vision  ·  ONNX  ·  No account or telemetry', 10,
                    MUTED, anchor='w').pack(fill='x', pady=(3, 0))

        row = self._card('RUN STATUS')
        self._label(row, '', 14, GREEN, anchor='w', textvariable=self.status).pack(fill='x')
        self._label(row, '', 9, MUTED, anchor='w', textvariable=self.details, wraplength=440,
                    justify='left').pack(fill='x', pady=3)
        self._label(row, '', 9, ACCENT, anchor='w', textvariable=self.stats).pack(fill='x')
        ctrls = tk.Frame(row, bg=SURFACE)
        ctrls.pack(fill='x', pady=(11, 0))
        tk.Button(ctrls, text='START VISION', command=self.start, bg=ACCENT, fg='#101724',
                  font=('Segoe UI', 10, 'bold'), relief='flat', padx=14, pady=8).pack(side='left')
        tk.Button(ctrls, text='STOP', command=self.stop, bg='#344153', fg=TEXT,
                  font=('Segoe UI', 10, 'bold'), relief='flat', padx=14, pady=8).pack(side='left', padx=8)

        inputs = self._card('INPUT SAFETY')
        tk.Checkbutton(inputs, text='Enable mouse movement  (only while holding right-click)',
                       variable=self.enabled, command=self._toggle_input,
                       bg=SURFACE, fg=TEXT, selectcolor=SURFACE,
                       activebackground=SURFACE, activeforeground=TEXT,
                       font=('Segoe UI', 10)).pack(anchor='w')
        tk.Checkbutton(inputs, text='Preview detections even when not aiming',
                       variable=self.preview, command=self._configure,
                       bg=SURFACE, fg=MUTED, selectcolor=SURFACE,
                       activebackground=SURFACE, activeforeground=MUTED).pack(anchor='w')
        tk.Checkbutton(inputs, text='Show click-through overlay',
                       variable=self.show_overlay, command=self._configure,
                       bg=SURFACE, fg=MUTED, selectcolor=SURFACE,
                       activebackground=SURFACE, activeforeground=MUTED).pack(anchor='w')
        self._label(inputs, 'F8 = emergency mouse-disable. Never sends clicks or keypresses.', 9,
                    RED, anchor='w').pack(fill='x', pady=(7, 0))

        model = self._card('MODEL & GAME WINDOW')
        field = tk.Frame(model, bg=SURFACE)
        field.pack(fill='x')
        tk.Entry(field, textvariable=self.model_path, bg='#293448', fg=TEXT, relief='flat',
                 insertbackground=TEXT, font=('Segoe UI', 9)).pack(side='left', fill='x', expand=True, ipady=6)
        tk.Button(field, text='Browse', command=self._browse, bg='#344153', fg=TEXT,
                  relief='flat', padx=10, pady=5).pack(side='left', padx=(7, 0))
        self._label(model, 'Foreground window title contains (separate with |):', 9,
                    MUTED, anchor='w').pack(fill='x', pady=(10, 4))
        tk.Entry(model, textvariable=self.window_title, bg='#293448', fg=TEXT,
                 relief='flat', insertbackground=TEXT).pack(fill='x', ipady=6)
        self.window_title.trace_add('write', lambda *_: self._configure())

        params = self._card('TARGETING PARAMETERS')
        self._slider(params, 'Detection confidence', self.confidence, .20, .85, .05)
        self._slider(params, 'Mouse strength', self.strength, .05, .65, .01)
        self._slider(params, 'Aim height (0=head, 1=feet)', self.aim_height, .10, .65, .01)
        self._slider(params, 'Detection area (pixels)', self.roi, 320, 720, 40)
        self._slider(params, 'Maximum mouse movement per frame', self.max_step, 4, 32, 2)

        foot = self._label(self.content,
                           'COCO person detector: cannot tell opponents from teammates.\n'
                           'Not tested or guaranteed in Black Ops 7; game rules still apply.',
                           9, MUTED, anchor='w', justify='left')
        foot.pack(fill='x', padx=20, pady=(1, 10))

    def _configure(self) -> None:
        self.engine.configure(window_titles=self.window_title.get(),
                              confidence=float(self.confidence.get()),
                              strength=float(self.strength.get()),
                              aim_height=float(self.aim_height.get()),
                              roi=int(self.roi.get()), max_step=int(self.max_step.get()),
                              overlay=bool(self.show_overlay.get()),
                              preview=bool(self.preview.get()))

    def _browse(self) -> None:
        path = filedialog.askopenfilename(title='Select ONNX person detector',
                                           filetypes=[('ONNX models', '*.onnx')])
        if path:
            self.model_path.set(path)
            self.engine.configure(model=path)
            if self.engine.is_running():
                messagebox.showinfo('Restart needed', 'Stop and start vision to load the selected model.')

    def _toggle_input(self) -> None:
        if self.enabled.get() and not self.engine.is_running():
            self.enabled.set(False)
            messagebox.showinfo('Vision is stopped', 'Start vision before enabling mouse movement.')
        self.engine.set_control(self.enabled.get())

    def start(self) -> None:
        self._configure()
        self.engine.configure(model=self.model_path.get())
        self.enabled.set(False)
        self.engine.start()

    def stop(self) -> None:
        self.engine.stop()
        self.enabled.set(False)
        self.overlay.draw(FrameState(), False)

    def _poll(self) -> None:
        snap = self.engine.snapshot()
        self.status.set(snap.state + (' · MOUSE ON' if snap.control_active else ''))
        self.details.set(snap.message)
        self.stats.set(f'Backend: {snap.provider}    {snap.fps:.0f} FPS    {snap.inference_ms:.1f} ms AI    People: {len(snap.boxes)}')
        if self.enabled.get() != snap.input_enabled and snap.state not in ('STOPPED', 'LOADING'):
            self.enabled.set(snap.input_enabled)
        self.overlay.draw(snap, self.show_overlay.get())
        self.root.after(65, self._poll)

    def close(self) -> None:
        self.engine.stop()
        self.overlay.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
