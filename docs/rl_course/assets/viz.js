/* ===========================================================================
   viz.js — 零依赖可视化与交互工具库（Canvas 2D + 一点 SVG）
   全部组件通过 data-viz 属性自动初始化，或手动调用 Viz.* 。
   =========================================================================== */
(function () {
  'use strict';

  const Viz = window.Viz = {};

  /* ---------- 基础数学工具 ---------- */
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const map = (x, a, b, c, d) => c + (d - c) * ((x - a) / (b - a));
  Viz.clamp = clamp; Viz.lerp = lerp; Viz.map = map;
  // 同时暴露到全局，方便各页内联脚本直接调用
  window.clamp = clamp; window.lerp = lerp; window.map = map;

  /* ---------- 主题颜色（从 CSS 变量读取，随深浅色切换） ---------- */
  function css(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  Viz.theme = () => ({
    text: css('--text', '#1d1d1f'),
    soft: css('--text-soft', '#515154'),
    faint: css('--text-faint', '#86868b'),
    accent: css('--accent', '#0071e3'),
    grid: css('--border', 'rgba(0,0,0,.1)'),
    green: css('--green', '#34c759'),
    orange: css('--orange', '#ff9f0a'),
    red: css('--red', '#ff3b30'),
    purple: css('--purple', '#af52de'),
    pink: css('--pink', '#ff2d55'),
    teal: css('--teal', '#30c8c9'),
    sunken: css('--bg-sunken', '#f5f5f7'),
    elev: css('--bg-elev', '#fff'),
  });

  /* ---------- 高 DPI Canvas ---------- */
  Viz.setupCanvas = function (canvas, cssW, cssH) {
    const dpr = window.devicePixelRatio || 1;
    // 若给定宽度用之，否则用容器宽度
    const w = cssW || canvas.clientWidth || canvas.parentElement.clientWidth;
    const h = cssH;
    canvas.style.height = h + 'px';
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    canvas._cssW = w; canvas._cssH = h;
    return ctx;
  };

  /* ---------- 绘图辅助 ---------- */
  Viz.axes = function (ctx, box, opts) {
    opts = opts || {};
    const t = Viz.theme();
    ctx.strokeStyle = t.faint; ctx.lineWidth = 1; ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.moveTo(box.x, box.y); ctx.lineTo(box.x, box.y + box.h);
    ctx.lineTo(box.x + box.w, box.y + box.h);
    ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillStyle = t.faint; ctx.font = '12px -apple-system, sans-serif';
    if (opts.xlabel) { ctx.textAlign = 'right'; ctx.fillText(opts.xlabel, box.x + box.w, box.y + box.h + 16); }
    if (opts.ylabel) { ctx.textAlign = 'left'; ctx.fillText(opts.ylabel, box.x - 2, box.y - 8); }
  };

  Viz.gridlines = function (ctx, box, nx, ny) {
    const t = Viz.theme();
    ctx.strokeStyle = t.grid; ctx.lineWidth = 1; ctx.globalAlpha = 0.6;
    for (let i = 0; i <= nx; i++) {
      const x = box.x + box.w * i / nx;
      ctx.beginPath(); ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); ctx.stroke();
    }
    for (let j = 0; j <= ny; j++) {
      const y = box.y + box.h * j / ny;
      ctx.beginPath(); ctx.moveTo(box.x, y); ctx.lineTo(box.x + box.w, y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  };

  Viz.plot = function (ctx, pts, color, width) {
    if (!pts.length) return;
    ctx.strokeStyle = color; ctx.lineWidth = width || 2.5;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.beginPath();
    pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.stroke();
  };

  Viz.fillCurve = function (ctx, pts, baseY, color, alpha) {
    if (!pts.length) return;
    ctx.save(); ctx.globalAlpha = alpha == null ? 0.14 : alpha;
    ctx.fillStyle = color; ctx.beginPath();
    ctx.moveTo(pts[0][0], baseY);
    pts.forEach(p => ctx.lineTo(p[0], p[1]));
    ctx.lineTo(pts[pts.length - 1][0], baseY); ctx.closePath(); ctx.fill();
    ctx.restore();
  };

  Viz.dot = function (ctx, x, y, r, color) {
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
  };

  Viz.text = function (ctx, str, x, y, color, font, align) {
    ctx.fillStyle = color; ctx.font = font || '13px -apple-system, sans-serif';
    ctx.textAlign = align || 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(str, x, y);
  };

  Viz.roundRect = function (ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  };

  /* ---------- 控件工厂：在容器里生成一个滑块，返回 input 元素 ---------- */
  Viz.slider = function (container, opts) {
    const wrap = document.createElement('div'); wrap.className = 'control';
    const label = document.createElement('label');
    const name = document.createElement('span'); name.innerHTML = opts.label;
    const val = document.createElement('span'); val.className = 'val';
    label.appendChild(name); label.appendChild(val);
    const input = document.createElement('input');
    input.type = 'range'; input.min = opts.min; input.max = opts.max;
    input.step = opts.step == null ? 0.01 : opts.step;
    input.value = opts.value;
    const fmt = opts.fmt || (v => (+v).toFixed(2));
    const sync = () => { val.textContent = fmt(input.value); };
    input.addEventListener('input', () => { sync(); opts.oninput && opts.oninput(+input.value); });
    sync();
    wrap.appendChild(label); wrap.appendChild(input);
    container.appendChild(wrap);
    return input;
  };

  Viz.button = function (container, text, onclick, ghost) {
    const b = document.createElement('button');
    b.className = 'btn' + (ghost ? ' ghost' : ''); b.textContent = text;
    b.addEventListener('click', onclick); container.appendChild(b); return b;
  };

  /* ---------- 极简 Python 语法高亮 ---------- */
  const PY_KW = new Set(('def class return if elif else for while in import from as with try except '
    + 'finally raise lambda yield None True False and or not is assert pass break continue global '
    + 'nonlocal del self super async await').split(' '));
  Viz.highlight = function () {
    document.querySelectorAll('pre code.lang-py, pre code.python').forEach(block => {
      if (block._hl) return; block._hl = true;
      const raw = block.textContent;
      const lines = raw.split('\n');
      const out = lines.map(line => {
        // 注释
        const ci = findComment(line);
        let code = ci >= 0 ? line.slice(0, ci) : line;
        let comment = ci >= 0 ? line.slice(ci) : '';
        code = code
          .replace(/(&|<|>)/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m]))
          .replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, m => `\u0001STR\u0001${m}\u0001/STR\u0001`)
          .replace(/\b(\d+\.?\d*)\b/g, '\u0001NUM\u0001$1\u0001/NUM\u0001')
          .replace(/\b([A-Za-z_]\w*)\b/g, (m) => PY_KW.has(m) ? `\u0001KW\u0001${m}\u0001/KW\u0001` : m)
          .replace(/\u0001STR\u0001/g, '<span class="tok-str">').replace(/\u0001\/STR\u0001/g, '</span>')
          .replace(/\u0001NUM\u0001/g, '<span class="tok-num">').replace(/\u0001\/NUM\u0001/g, '</span>')
          .replace(/\u0001KW\u0001/g, '<span class="tok-kw">').replace(/\u0001\/KW\u0001/g, '</span>');
        if (comment) comment = '<span class="tok-com">' + comment.replace(/[&<>]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m])) + '</span>';
        return code + comment;
      });
      block.innerHTML = out.join('\n');
    });
    function findComment(line) {
      let inStr = null;
      for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (inStr) { if (c === inStr && line[i - 1] !== '\\') inStr = null; }
        else if (c === '"' || c === "'") inStr = c;
        else if (c === '#') return i;
      }
      return -1;
    }
  };

  /* ---------- TOC 目录滚动高亮 ---------- */
  Viz.scrollspy = function () {
    const links = Array.from(document.querySelectorAll('.toc a'));
    if (!links.length) return;
    const map = new Map();
    links.forEach(a => { const id = a.getAttribute('href').slice(1); const el = document.getElementById(id); if (el) map.set(el, a); });
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          links.forEach(l => l.classList.remove('active'));
          const a = map.get(e.target); if (a) a.classList.add('active');
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px' });
    map.forEach((a, el) => obs.observe(el));
  };

  /* ---------- 自动从 h2[id] 生成 TOC ---------- */
  Viz.buildTOC = function () {
    const toc = document.querySelector('.toc'); if (!toc) return;
    const hs = document.querySelectorAll('.content h2[id]');
    if (!hs.length) return;
    let html = '<div class="toc-h">本页目录</div>';
    hs.forEach(h => { html += `<a href="#${h.id}">${h.textContent}</a>`; });
    toc.innerHTML = html;
  };

  /* ---------- 重绘登记：深浅色切换时刷新所有可视化 ---------- */
  const redrawers = [];
  Viz.onRedraw = fn => { redrawers.push(fn); fn(); };
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      redrawers.forEach(fn => { try { fn(); } catch (e) {} });
    });
  }
  let rT;
  window.addEventListener('resize', () => { clearTimeout(rT); rT = setTimeout(() => redrawers.forEach(fn => { try { fn(); } catch (e) {} }), 150); });

  /* ---------- 代码导读：train.py 调用链 SVG ---------- */
  Viz.callChainTrain = function (hostId, outId) {
    const steps = [
      ['CLI', 'python train.py --task ...', '你敲的命令行参数'],
      ['AppLauncher', '启动 Isaac Sim', '必须在 import isaaclab 模块之前'],
      ['hydra_task_config', '加载 env_cfg + ppo_cfg.yaml', '根据 gym 注册信息解析配置'],
      ['gym.make', '实例化 XhandWaveInhandRotateEnv', '创建 num_envs 个并行仿真'],
      ['GymStyleEnvWrapper', '适配 VecEnv 接口', 'clip action、自动 reset'],
      ['PPO(...)', '建 ActorCritic + Buffer', 'eval("PPO") 或 ProprioAdapt'],
      ['restore_train', '加载 checkpoint', 'ProprioAdapt 必须加载 stage1'],
      ['agent.train()', '主训练循环', '直到 max_agent_steps'],
    ];
    const A = '#0071e3', O = '#ff9f0a', G = '#34c759';
    let svg = '<svg viewBox="0 0 720 ' + (steps.length * 52 + 20) + '" role="img"><defs><marker id="ahTrain" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="var(--text-faint)"></path></marker></defs>';
    steps.forEach((s, i) => {
      const y = 10 + i * 52;
      const fill = i === 0 ? O : i === steps.length - 1 ? G : A;
      svg += '<g class="chain-box" data-i="' + i + '" style="cursor:pointer"><rect x="20" y="' + y + '" width="680" height="42" rx="10" fill="' + fill + '" opacity="0.92"></rect>';
      svg += '<text x="36" y="' + (y + 26) + '" font-size="13" font-weight="600" fill="#fff">' + (i + 1) + '. ' + s[0] + '</text></g>';
      if (i < steps.length - 1) svg += '<path d="M360,' + (y + 42) + ' L360,' + (y + 52) + '" stroke="var(--text-faint)" stroke-width="2" marker-end="url(#ahTrain)"></path>';
    });
    svg += '</svg>';
    const host = document.getElementById(hostId);
    const out = document.getElementById(outId);
    if (!host) return;
    host.innerHTML = svg;
    host.querySelectorAll('.chain-box').forEach(g => {
      g.addEventListener('mouseenter', () => {
        const s = steps[+g.dataset.i];
        if (out && s) out.innerHTML = '<b>' + s[0] + '：</b> ' + s[1] + ' — ' + s[2];
      });
    });
  };

  /* ---------- 环境 step 时序 SVG ---------- */
  Viz.envStepSequence = function (hostId) {
    const items = ['_pre_physics_step', 'physics×decimation', '_apply_action', '_get_observations', '_get_rewards', '_get_dones'];
    let svg = '<svg viewBox="0 0 600 320" role="img"><text x="20" y="24" font-size="13" font-weight="700" fill="var(--accent)">DirectRLEnv 一步 control step</text>';
    items.forEach((label, i) => {
      const x = 20 + (i % 3) * 190, y = 50 + Math.floor(i / 3) * 90;
      svg += '<rect x="' + x + '" y="' + y + '" width="170" height="48" rx="12" fill="var(--bg-sunken)" stroke="var(--border-strong)"></rect>';
      svg += '<text x="' + (x + 85) + '" y="' + (y + 30) + '" text-anchor="middle" font-size="12" font-weight="600" fill="var(--text)">' + label + '</text>';
    });
    svg += '</svg>';
    const host = document.getElementById(hostId);
    if (host) host.innerHTML = svg;
  };

  /* ---------- 配置优先级栈 ---------- */
  Viz.configPriorityStack = function (hostId, outId) {
    const layers = [
      ['CLI 命令行', '最高', '--num_envs 10 覆盖 yaml 里的 16384'],
      ['Hydra 运行时', '中', '@hydra_task_config 注入 env/agent cfg'],
      ['ppo_cfg.yaml', '低', '默认算法超参'],
      ['xhand_wave_env_cfg.py', '低', '默认环境/奖励/随机化'],
    ];
    let svg = '<svg viewBox="0 0 400 220" role="img">';
    layers.forEach((l, i) => {
      const y = 180 - i * 48, w = 280 + i * 20, x = 60 - i * 10;
      const op = 0.55 + i * 0.12;
      svg += '<g class="cfg-layer" data-i="' + i + '" style="cursor:pointer"><rect x="' + x + '" y="' + y + '" width="' + w + '" height="40" rx="10" fill="var(--accent)" opacity="' + op + '"></rect>';
      svg += '<text x="' + (x + w / 2) + '" y="' + (y + 25) + '" text-anchor="middle" font-size="13" font-weight="600" fill="#fff">' + l[0] + ' (' + l[1] + ')</text></g>';
    });
    svg += '</svg>';
    const host = document.getElementById(hostId);
    const out = document.getElementById(outId);
    if (!host) return;
    host.innerHTML = svg;
    host.querySelectorAll('.cfg-layer').forEach(g => {
      g.addEventListener('mouseenter', () => {
        const l = layers[+g.dataset.i];
        if (out && l) out.innerHTML = '<b>' + l[0] + '（优先级' + l[1] + '）。</b> ' + l[2];
      });
    });
  };

  /* ---------- 入口 ---------- */
  document.addEventListener('DOMContentLoaded', function () {
    Viz.highlight();
    Viz.buildTOC();
    Viz.scrollspy();
  });
})();
