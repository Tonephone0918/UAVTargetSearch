from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .baselines import get_baseline_action_selector
from .config import ExperimentConfig
from .env import UAVSearchEnv
from .runtime import (
    apply_env_overrides,
    build_mappo_from_env,
    build_policy_from_env,
    config_from_dict,
    load_checkpoint_module,
    load_checkpoint_policy,
    resolve_device,
)


def _snapshot(
    env: UAVSearchEnv,
    step_reward: float,
    total_reward: float,
    info: Dict[str, float],
) -> Dict:
    global_tpm = env._global_tpm()
    global_dpm = np.mean([m.dpm for m in env.maps], axis=0)
    targets = []
    for i, t in enumerate(env.targets):
        targets.append(
            {
                "x": float(t.x),
                "y": float(t.y),
                "found": i in env.found_targets,
            }
        )

    return {
        "t": int(env.t),
        "uavs": [[int(x), int(y), int(z)] for x, y, z in env.uavs],
        "threats": [[int(x), int(y)] for x, y in env.threats],
        "targets": targets,
        "tpm": global_tpm.astype(float).tolist(),
        "dpm": global_dpm.astype(float).tolist(),
        "coverage": env.coverage.astype(int).tolist(),
        "step_reward": float(step_reward),
        "total_reward": float(total_reward),
        "search_rate": float(info.get("search_rate", 0.0)),
        "coverage_rate": float(info.get("coverage_rate", float(env.coverage.mean()))),
        "collisions": float(info.get("collisions", 0.0)),
        "error_rate": float(info.get("error_rate", 0.0)),
    }


def _collect_with_action_selector(
    env: UAVSearchEnv,
    action_selector,
    max_steps: int | None = None,
) -> Dict:
    obs = env.reset()
    done = False
    total_reward = 0.0
    frames: List[Dict] = [_snapshot(env, step_reward=0.0, total_reward=0.0, info={})]

    while not done:
        acts = action_selector(obs)
        obs, reward, done, info = env.step(acts)
        total_reward += reward
        frames.append(_snapshot(env, step_reward=reward, total_reward=total_reward, info=info))

        if max_steps is not None and env.t >= max_steps:
            done = True

    summary = {
        "steps": int(env.t),
        "total_reward": float(total_reward),
        "found_targets": int(len(env.found_targets)),
        "all_targets": int(env.cfg.n_targets),
        "final_search_rate": float(frames[-1]["search_rate"]) if frames else 0.0,
        "final_coverage_rate": float(frames[-1]["coverage_rate"]) if frames else 0.0,
    }
    return {"summary": summary, "frames": frames}


@torch.no_grad()
def collect_rollout(
    checkpoint_path: str,
    device: str = "auto",
    max_steps: int | None = None,
    env_overrides: Dict | None = None,
) -> Dict:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_device = resolve_device(device)
    ckpt = torch.load(ckpt_path, map_location=run_device)
    algo = ckpt.get("algo", "hrvdn")

    cfg_raw = ckpt.get("config", {})
    cfg = config_from_dict(cfg_raw)
    cfg.reward.mode = ckpt.get("reward_mode", cfg.reward.mode)
    apply_env_overrides(cfg, **(env_overrides or {}))

    env = UAVSearchEnv(cfg.env, cfg.reward, seed=cfg.train.seed)
    if algo == "mappo":
        actor, critic = build_mappo_from_env(cfg, env, run_device)
        load_checkpoint_module(actor, ckpt["actor_state_dict"], checkpoint_path, "MAPPO actor")
        actor.eval()
        critic.eval()

        def action_selector(obs):
            maps = torch.tensor(np.stack([o["map"] for o in obs]), dtype=torch.float32, device=run_device)
            extras = torch.tensor(np.stack([o["extra"] for o in obs]), dtype=torch.float32, device=run_device)
            masks = torch.tensor(np.stack([o["action_mask"] for o in obs]), dtype=torch.bool, device=run_device)
            logits = actor(maps, extras).masked_fill(~masks, -1e9)
            return logits.argmax(dim=-1).tolist()

    else:
        policy = build_policy_from_env(cfg, env, run_device)
        load_checkpoint_policy(policy, ckpt["policy_state_dict"], checkpoint_path)
        policy.eval()

        hs = [torch.zeros(1, 1, policy.gru.hidden_size, device=run_device) for _ in range(env.cfg.n_uavs)]

        def action_selector(obs):
            nonlocal hs
            acts = []
            for i, o in enumerate(obs):
                om = torch.tensor(o["map"], dtype=torch.float32, device=run_device).flatten().unsqueeze(0)
                ex = torch.tensor(o["extra"], dtype=torch.float32, device=run_device).unsqueeze(0)
                q, hs[i] = policy(om, ex, hs[i])
                mask = torch.tensor(o["action_mask"], dtype=torch.bool, device=run_device).unsqueeze(0)
                q = q.masked_fill(~mask, -1e9)
                acts.append(int(q.argmax(dim=-1).item()))
            return acts

    rollout = _collect_with_action_selector(env, action_selector, max_steps=max_steps)

    return {
        "meta": {
            "source": str(ckpt_path),
            "policy": algo,
            "map_size": int(env.cfg.map_size),
            "n_uavs": int(env.cfg.n_uavs),
            "n_altitudes": int(env.cfg.n_altitudes),
            "n_targets": int(env.cfg.n_targets),
            "n_threats": int(env.cfg.n_threats),
            "max_steps": int(env.cfg.max_steps),
            "reward_mode": str(cfg.reward.mode),
        },
        "summary": rollout["summary"],
        "frames": rollout["frames"],
    }


def collect_baseline_rollout(
    cfg: ExperimentConfig,
    baseline: str = "greedy",
    max_steps: int | None = None,
) -> Dict:
    env = UAVSearchEnv(cfg.env, cfg.reward, seed=cfg.train.seed)
    action_selector = get_baseline_action_selector(baseline)
    rollout = _collect_with_action_selector(env, action_selector, max_steps=max_steps)

    return {
        "meta": {
            "source": f"baseline:{baseline}",
            "policy": baseline,
            "map_size": int(env.cfg.map_size),
            "n_uavs": int(env.cfg.n_uavs),
            "n_altitudes": int(env.cfg.n_altitudes),
            "n_targets": int(env.cfg.n_targets),
            "n_threats": int(env.cfg.n_threats),
            "max_steps": int(env.cfg.max_steps),
            "reward_mode": str(cfg.reward.mode),
        },
        "summary": rollout["summary"],
        "frames": rollout["frames"],
    }


def save_rollout_html(rollout: Dict, output_html: str) -> Path:
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rollout, ensure_ascii=True)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HRVDN Dynamic Search Replay</title>
  <style>
    body {{
      margin: 0;
      padding: 20px;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f4f7fb;
      color: #0f172a;
    }}
    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #dbe3ee;
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 14px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .title {{
      margin: 0 0 8px 0;
      font-size: 20px;
    }}
    .muted {{
      margin: 0;
      color: #475569;
      font-size: 13px;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .viz-row {{
      display: flex;
      gap: 14px;
      align-items: flex-start;
      justify-content: center;
      flex-wrap: wrap;
    }}
    .side-panel {{
      width: 180px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .side-card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 10px;
    }}
    .side-title {{
      margin: 0 0 8px 0;
      font-size: 13px;
      color: #334155;
      font-weight: 600;
    }}
    .mini-canvas {{
      width: 100%;
      aspect-ratio: 1 / 1;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #ffffff;
      display: block;
      margin: 0;
    }}
    .colorbar {{
      width: 26px;
      height: 150px;
      border-radius: 8px;
      border: 1px solid #cbd5e1;
      flex: 0 0 auto;
    }}
    .colorbar-tpm {{
      background: linear-gradient(180deg, #15803d 0%, #1e40af 100%);
    }}
    .colorbar-dpm {{
      background: linear-gradient(180deg, #16a34a 0%, #0ea5e9 100%);
    }}
    .colorbar-wrap {{
      display: flex;
      gap: 10px;
      align-items: stretch;
    }}
    .colorbar-labels {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      font-size: 12px;
      color: #475569;
      min-width: 62px;
    }}
    button, select {{
      border: 1px solid #cbd5e1;
      background: #fff;
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 14px;
      cursor: pointer;
    }}
    button:hover {{
      background: #f8fafc;
    }}
    input[type="range"] {{
      width: 420px;
      max-width: 100%;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      font-size: 14px;
    }}
    .stat {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 8px 10px;
    }}
    canvas {{
      width: 100%;
      height: auto;
      max-width: 880px;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      background: #ffffff;
      display: block;
      margin: 10px auto 4px auto;
    }}
    .legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 13px;
      color: #475569;
      margin-top: 8px;
    }}
    .tag {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      display: inline-block;
      flex: 0 0 auto;
    }}
    .swatch-circle {{
      border-radius: 50%;
    }}
    .swatch-square {{
      border-radius: 3px;
    }}
    .swatch-triangle {{
      width: 0;
      height: 0;
      border-left: 7px solid transparent;
      border-right: 7px solid transparent;
      border-bottom: 12px solid #ef4444;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1 class="title">HRVDN Dynamic Search Replay</h1>
      <p class="muted" id="meta"></p>
    </div>

    <div class="panel">
      <div class="controls">
        <button id="playBtn">Play</button>
        <button id="resetBtn">Reset</button>
        <label>Speed
          <select id="speedSel">
            <option value="1">1x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
            <option value="8">8x</option>
          </select>
        </label>
        <input id="frameSlider" type="range" min="0" value="0" step="1" />
        <span id="frameText"></span>
      </div>

      <div class="viz-row">
        <canvas id="cv" width="880" height="880"></canvas>
        <div class="side-panel">
          <div class="side-card">
            <p class="side-title">TPM Color Scale</p>
            <div class="colorbar-wrap">
              <div class="colorbar colorbar-tpm"></div>
              <div class="colorbar-labels">
                <span>high target prob.</span>
                <span>medium</span>
                <span>low target prob.</span>
              </div>
            </div>
          </div>
          <div class="side-card">
            <p class="side-title">DPM Heatmap</p>
            <canvas id="dpmCv" class="mini-canvas" width="240" height="240"></canvas>
          </div>
          <div class="side-card">
            <p class="side-title">DPM Color Scale</p>
            <div class="colorbar-wrap">
              <div class="colorbar colorbar-dpm"></div>
              <div class="colorbar-labels">
                <span>high revisit</span>
                <span>medium</span>
                <span>low revisit</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="legend">
        <span class="tag"><span class="swatch swatch-square" style="background:#16a34a"></span>TPM high</span>
        <span class="tag"><span class="swatch swatch-square" style="background:#16a34a"></span>DPM high</span>
        <span class="tag"><span class="swatch swatch-square" style="background:#cfe8ff"></span>Coverage</span>
        <span class="tag"><span class="swatch swatch-square" style="background:#9ca3af"></span>Obstacle</span>
        <span class="tag"><span class="swatch swatch-triangle" style="border-bottom-color:#ef4444"></span>Target (unfound)</span>
        <span class="tag"><span class="swatch swatch-triangle" style="border-bottom-color:#22c55e"></span>Target (found)</span>
        <span class="tag"><span class="swatch swatch-circle" style="background:#f59e0b"></span>UAV low altitude</span>
        <span class="tag"><span class="swatch swatch-circle" style="background:#ec4899"></span>UAV mid altitude</span>
        <span class="tag"><span class="swatch swatch-circle" style="background:#ef4444"></span>UAV high altitude</span>
        <span class="tag"><span class="swatch swatch-circle" style="background:#2563eb"></span>UAV path</span>
      </div>
    </div>

    <div class="panel">
      <div class="stats" id="stats"></div>
    </div>
  </div>

  <script>
    const DATA = {payload};
    const pathColors = [
      "#2563eb", "#0891b2", "#16a34a", "#e11d48", "#7c3aed",
      "#ea580c", "#0f766e", "#4f46e5", "#059669", "#d946ef"
    ];
    const altitudeColors = ["#f97316", "#ec4899", "#dc2626"];

    const cv = document.getElementById("cv");
    const ctx = cv.getContext("2d");
    const dpmCv = document.getElementById("dpmCv");
    const dpmCtx = dpmCv.getContext("2d");
    const slider = document.getElementById("frameSlider");
    const playBtn = document.getElementById("playBtn");
    const resetBtn = document.getElementById("resetBtn");
    const speedSel = document.getElementById("speedSel");
    const frameText = document.getElementById("frameText");
    const stats = document.getElementById("stats");
    const meta = document.getElementById("meta");

    meta.textContent =
      "source: " + DATA.meta.source +
      " | policy: " + DATA.meta.policy +
      " | map: " + DATA.meta.map_size + "x" + DATA.meta.map_size +
      " | uavs: " + DATA.meta.n_uavs +
      " | altitudes: " + DATA.meta.n_altitudes +
      " | targets: " + DATA.meta.n_targets +
      " | threats: " + DATA.meta.n_threats +
      " | reward_mode: " + DATA.meta.reward_mode;

    let idx = 0;
    let timer = null;
    slider.max = Math.max(0, DATA.frames.length - 1);

    function cellSize() {{
      return cv.width / DATA.meta.map_size;
    }}

    function drawGrid() {{
      const n = DATA.meta.map_size;
      const cs = cellSize();
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      for (let i = 0; i <= n; i++) {{
        const p = i * cs;
        ctx.beginPath();
        ctx.moveTo(0, p);
        ctx.lineTo(cv.width, p);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(p, 0);
        ctx.lineTo(p, cv.height);
        ctx.stroke();
      }}
    }}

    function tpmColor(prob) {{
      const p = Math.max(0, Math.min(1, prob));
      const r = Math.round(30 - p * 9);
      const g = Math.round(64 + p * 64);
      const b = Math.round(175 - p * 114);
      return "rgb(" + r + "," + g + "," + b + ")";
    }}

    function dpmMaxValue() {{
      let vmax = 0;
      for (const frame of DATA.frames) {{
        for (const row of frame.dpm) {{
          for (const val of row) {{
            if (val > vmax) vmax = val;
          }}
        }}
      }}
      return Math.max(1e-6, vmax);
    }}

    const DPM_MAX = dpmMaxValue();

    function dpmColor(value) {{
      const p = Math.max(0, Math.min(1, value / DPM_MAX));
      const r = Math.round(14 + p * 8);
      const g = Math.round(165 - p * 2);
      const b = Math.round(233 - p * 103);
      return "rgb(" + r + "," + g + "," + b + ")";
    }}

    function drawTpm(frame) {{
      const n = DATA.meta.map_size;
      const cs = cellSize();
      for (let x = 0; x < n; x++) {{
        for (let y = 0; y < n; y++) {{
          ctx.fillStyle = tpmColor(frame.tpm[x][y]);
          ctx.fillRect(y * cs, x * cs, cs, cs);
        }}
      }}
    }}

    function drawCoverage(frame) {{
      const n = DATA.meta.map_size;
      const cs = cellSize();
      for (let x = 0; x < n; x++) {{
        for (let y = 0; y < n; y++) {{
          if (frame.coverage[x][y]) {{
            ctx.fillStyle = "rgba(207, 232, 255, 0.18)";
            ctx.fillRect(y * cs, x * cs, cs, cs);
            ctx.strokeStyle = "rgba(59, 130, 246, 0.28)";
            ctx.lineWidth = Math.max(1, cs * 0.03);
            ctx.strokeRect(y * cs + 0.5, x * cs + 0.5, cs - 1, cs - 1);
          }}
        }}
      }}
    }}

    function drawDpm(frame) {{
      const n = DATA.meta.map_size;
      const cs = dpmCv.width / n;
      dpmCtx.clearRect(0, 0, dpmCv.width, dpmCv.height);
      for (let x = 0; x < n; x++) {{
        for (let y = 0; y < n; y++) {{
          dpmCtx.fillStyle = dpmColor(frame.dpm[x][y]);
          dpmCtx.fillRect(y * cs, x * cs, cs, cs);
        }}
      }}
      dpmCtx.strokeStyle = "rgba(148, 163, 184, 0.45)";
      dpmCtx.lineWidth = 1;
      for (let i = 0; i <= n; i++) {{
        const p = i * cs;
        dpmCtx.beginPath();
        dpmCtx.moveTo(0, p);
        dpmCtx.lineTo(dpmCv.width, p);
        dpmCtx.stroke();
        dpmCtx.beginPath();
        dpmCtx.moveTo(p, 0);
        dpmCtx.lineTo(p, dpmCv.height);
        dpmCtx.stroke();
      }}
    }}

    function drawThreats(frame) {{
      const cs = cellSize();
      for (const th of frame.threats) {{
        const x = th[0], y = th[1];
        ctx.fillStyle = "#9ca3af";
        ctx.strokeStyle = "#6b7280";
        ctx.lineWidth = Math.max(1, cs * 0.04);
        ctx.beginPath();
        ctx.roundRect(y * cs + cs * 0.12, x * cs + cs * 0.12, cs * 0.76, cs * 0.76, cs * 0.08);
        ctx.fill();
        ctx.stroke();
      }}
    }}

    function drawTriangle(cx, cy, size, fill, stroke) {{
      ctx.beginPath();
      ctx.moveTo(cx, cy - size);
      ctx.lineTo(cx - size * 0.9, cy + size * 0.75);
      ctx.lineTo(cx + size * 0.9, cy + size * 0.75);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = Math.max(1.5, size * 0.16);
      ctx.stroke();
    }}

    function drawTargets(frame) {{
      const cs = cellSize();
      for (const tg of frame.targets) {{
        const fill = tg.found ? "#22c55e" : "#ef4444";
        const stroke = tg.found ? "#166534" : "#991b1b";
        drawTriangle((tg.y + 0.5) * cs, (tg.x + 0.5) * cs, cs * 0.3, fill, stroke);
      }}
    }}

    function drawPaths(frameIdx) {{
      const cs = cellSize();
      const n = DATA.meta.n_uavs;
      for (let i = 0; i < n; i++) {{
        ctx.strokeStyle = pathColors[i % pathColors.length];
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        for (let t = 0; t <= frameIdx; t++) {{
          const p = DATA.frames[t].uavs[i];
          const x = (p[1] + 0.5) * cs;
          const y = (p[0] + 0.5) * cs;
          if (!started) {{
            ctx.moveTo(x, y);
            started = true;
          }} else {{
            ctx.lineTo(x, y);
          }}
        }}
        ctx.stroke();
      }}
    }}

    function getUavColor(altitude) {{
      if (DATA.meta.n_altitudes <= 1) return altitudeColors[0];
      const scaled = Math.round(
        (Math.max(0, altitude) / Math.max(1, DATA.meta.n_altitudes - 1)) * (altitudeColors.length - 1)
      );
      const idx = Math.max(0, Math.min(altitudeColors.length - 1, scaled));
      return altitudeColors[idx];
    }}

    function getUavHeading(frameIdx, uavIdx) {{
      for (let t = frameIdx; t > 0; t--) {{
        const curr = DATA.frames[t].uavs[uavIdx];
        const prev = DATA.frames[t - 1].uavs[uavIdx];
        const dx = curr[1] - prev[1];
        const dy = curr[0] - prev[0];
        if (dx !== 0 || dy !== 0) {{
          return Math.atan2(dy, dx);
        }}
      }}
      return -Math.PI / 2;
    }}

    function mixColor(color, alpha) {{
      const hex = color.replace("#", "");
      const raw = hex.length === 3
        ? hex.split("").map((ch) => ch + ch).join("")
        : hex;
      const r = parseInt(raw.slice(0, 2), 16);
      const g = parseInt(raw.slice(2, 4), 16);
      const b = parseInt(raw.slice(4, 6), 16);
      const blend = (channel) => Math.round(channel * (1 - alpha) + 255 * alpha);
      return "rgb(" + blend(r) + "," + blend(g) + "," + blend(b) + ")";
    }}

    function drawQuadcopter(x, y, size, color, label, heading) {{
      const rotorOffset = size * 0.78;
      const rotorRadius = size * 0.24;
      const armWidth = Math.max(2.5, size * 0.16);
      const rotorColor = mixColor(color, 0.18);

      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(heading);

      ctx.strokeStyle = color;
      ctx.lineWidth = armWidth;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(-rotorOffset, -rotorOffset);
      ctx.lineTo(rotorOffset, rotorOffset);
      ctx.moveTo(-rotorOffset, rotorOffset);
      ctx.lineTo(rotorOffset, -rotorOffset);
      ctx.stroke();

      const rotors = [
        [-rotorOffset, -rotorOffset],
        [-rotorOffset, rotorOffset],
        [rotorOffset, -rotorOffset],
        [rotorOffset, rotorOffset],
      ];

      for (const rotor of rotors) {{
        ctx.fillStyle = rotorColor;
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = Math.max(1.5, size * 0.07);
        ctx.beginPath();
        ctx.arc(rotor[0], rotor[1], rotorRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }}

      ctx.fillStyle = color;
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = Math.max(1.5, size * 0.09);
      ctx.beginPath();
      ctx.roundRect(-size * 0.4, -size * 0.3, size * 0.8, size * 0.6, size * 0.18);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = mixColor(color, 0.35);
      ctx.beginPath();
      ctx.moveTo(size * 0.2, 0);
      ctx.lineTo(size * 0.02, -size * 0.12);
      ctx.lineTo(size * 0.02, size * 0.12);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = "#111111";
      ctx.beginPath();
      ctx.moveTo(size * 1.02, 0);
      ctx.lineTo(size * 0.58, -size * 0.14);
      ctx.lineTo(size * 0.58, size * 0.14);
      ctx.closePath();
      ctx.fill();

      ctx.restore();

      ctx.fillStyle = "#0f172a";
      ctx.font = Math.max(10, size * 0.5) + "px Segoe UI";
      ctx.fillText(String(label), x + size * 0.95, y - size * 0.75);
    }}

    function drawUavs(frame, frameIdx) {{
      const cs = cellSize();
      for (let i = 0; i < frame.uavs.length; i++) {{
        const u = frame.uavs[i];
        const x = (u[1] + 0.5) * cs;
        const y = (u[0] + 0.5) * cs;
        const color = getUavColor(u[2]);
        const heading = getUavHeading(frameIdx, i);
        drawQuadcopter(x, y, cs * 0.34, color, i, heading);
      }}
    }}

    function renderStats(frame) {{
      const rows = [
        ["step", String(frame.t)],
        ["step_reward", frame.step_reward.toFixed(3)],
        ["total_reward", frame.total_reward.toFixed(3)],
        ["search_rate", frame.search_rate.toFixed(3)],
        ["coverage_rate", frame.coverage_rate.toFixed(3)],
        ["collisions", frame.collisions.toFixed(0)],
        ["error_rate", frame.error_rate.toFixed(4)],
      ];
      stats.innerHTML = rows
        .map(([k, v]) => `<div class="stat"><strong>${{k}}</strong>: ${{v}}</div>`)
        .join("");
    }}

    function render() {{
      const frame = DATA.frames[idx];
      frameText.textContent = `${{idx}} / ${{DATA.frames.length - 1}}`;
      slider.value = String(idx);

      ctx.clearRect(0, 0, cv.width, cv.height);
      drawTpm(frame);
      drawCoverage(frame);
      drawPaths(idx);
      drawTargets(frame);
      drawThreats(frame);
      drawGrid();
      drawUavs(frame, idx);
      drawDpm(frame);
      renderStats(frame);
    }}

    function play() {{
      if (timer) return;
      playBtn.textContent = "Pause";
      timer = setInterval(() => {{
        const speed = Number(speedSel.value || "1");
        idx = Math.min(DATA.frames.length - 1, idx + speed);
        render();
        if (idx >= DATA.frames.length - 1) stop();
      }}, 250);
    }}

    function stop() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
      }}
      playBtn.textContent = "Play";
    }}

    playBtn.addEventListener("click", () => {{
      if (timer) stop();
      else play();
    }});
    resetBtn.addEventListener("click", () => {{
      stop();
      idx = 0;
      render();
    }});
    slider.addEventListener("input", () => {{
      idx = Number(slider.value);
      render();
    }});

    render();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def generate_rollout_html(
    checkpoint_path: str,
    output_html: str = "runs/hrvdn/search_replay.html",
    device: str = "auto",
    max_steps: int | None = None,
    env_overrides: Dict | None = None,
) -> Path:
    rollout = collect_rollout(
        checkpoint_path=checkpoint_path,
        device=device,
        max_steps=max_steps,
        env_overrides=env_overrides,
    )
    return save_rollout_html(rollout, output_html)


def generate_baseline_rollout_html(
    cfg: ExperimentConfig,
    baseline: str = "greedy",
    output_html: str = "runs/hrvdn/greedy_search_replay.html",
    max_steps: int | None = None,
) -> Path:
    rollout = collect_baseline_rollout(cfg=cfg, baseline=baseline, max_steps=max_steps)
    return save_rollout_html(rollout, output_html)
