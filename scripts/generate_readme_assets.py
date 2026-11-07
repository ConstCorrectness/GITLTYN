"""Generate publication-grade SVG assets for GITLTYN README."""
import os

def generate_architecture_svg(out_path: str):
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 480" width="100%" height="100%" style="background:#0f172a; border-radius:12px; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <defs>
    <linearGradient id="primaryGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#10b981" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#059669" stop-opacity="0.05"/>
    </linearGradient>
    <linearGradient id="probeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#6366f1" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#4f46e5" stop-opacity="0.05"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#94a3b8"/>
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#10b981"/>
    </marker>
    <marker id="arrow-indigo" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#818cf8"/>
    </marker>
  </defs>

  <!-- Title & Subtitle -->
  <text x="480" y="38" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="700" letter-spacing="0.5">GITLTYN SYSTEM ARCHITECTURE</text>
  <text x="480" y="58" text-anchor="middle" fill="#94a3b8" font-size="12">Continuous Latent Looping with Detachable Ring-Buffer Latent Tee &amp; Stop-Gradient Sidecar Probe</text>

  <!-- Phase 1: Input & Prompt Embedding -->
  <g transform="translate(40, 95)">
    <rect width="180" height="130" rx="10" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>
    <text x="90" y="28" text-anchor="middle" fill="#38bdf8" font-size="13" font-weight="600">Prompt Encoding</text>
    <text x="90" y="52" text-anchor="middle" fill="#cbd5e1" font-size="11">Input Token Sequence x</text>
    <rect x="20" y="68" width="140" height="42" rx="6" fill="#0f172a" stroke="#475569" stroke-width="1"/>
    <text x="90" y="86" text-anchor="middle" fill="#e2e8f0" font-size="11" font-family="monospace">Qwen3 Backbone</text>
    <text x="90" y="100" text-anchor="middle" fill="#94a3b8" font-size="10">Top-Layer h_t^(L) ∈ ℝᵈ</text>
  </g>

  <!-- Connection to Latent Loop -->
  <line x1="220" y1="160" x2="275" y2="160" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="248" y="150" text-anchor="middle" fill="#94a3b8" font-size="10">z₀ = h_t^(L)</text>

  <!-- Phase 2: Continuous Latent Loop Core -->
  <g transform="translate(280, 85)">
    <rect width="400" height="150" rx="12" fill="url(#primaryGrad)" stroke="#10b981" stroke-width="2"/>
    <rect x="15" y="12" width="170" height="22" rx="4" fill="#065f46"/>
    <text x="100" y="27" text-anchor="middle" fill="#a7f3d0" font-size="11" font-weight="700">CONTINUOUS LATENT LOOP</text>
    <text x="320" y="27" text-anchor="middle" fill="#10b981" font-size="11" font-weight="600">Bypasses W_unembed (k = 1..K)</text>

    <!-- Adapter Box -->
    <rect x="25" y="48" width="350" height="74" rx="8" fill="#1e293b" stroke="#059669" stroke-width="1.5"/>
    <text x="200" y="72" text-anchor="middle" fill="#f1f5f9" font-size="13" font-weight="600">Gated Latent Residual Adapter f_θ(z)</text>
    <text x="200" y="92" text-anchor="middle" fill="#a7f3d0" font-size="11" font-family="monospace">z_k = RMSNorm( z_{k-1} + SwiGLU(z_{k-1}) )</text>
    <text x="200" y="110" text-anchor="middle" fill="#94a3b8" font-size="10">Pure ℝᵈ Dynamics • Zero Token KV-Cache Growth</text>
  </g>

  <!-- Recurrence Arrow on Latent Loop -->
  <path d="M 520 85 C 520 60, 440 60, 440 85" fill="none" stroke="#10b981" stroke-width="2" marker-end="url(#arrow-green)"/>
  <text x="480" y="55" text-anchor="middle" fill="#10b981" font-size="10" font-weight="600">Repeat K Steps</text>

  <!-- Connection to Final Emission -->
  <line x1="680" y1="160" x2="735" y2="160" stroke="#10b981" stroke-width="2" marker-end="url(#arrow-green)"/>
  <text x="708" y="150" text-anchor="middle" fill="#10b981" font-size="10">z_K</text>

  <!-- Phase 3: Single Terminal Emission -->
  <g transform="translate(740, 95)">
    <rect width="180" height="130" rx="10" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>
    <text x="90" y="28" text-anchor="middle" fill="#f59e0b" font-size="13" font-weight="600">Terminal Emission</text>
    <text x="90" y="48" text-anchor="middle" fill="#94a3b8" font-size="10">Invoked Exactly Once</text>
    <rect x="20" y="58" width="140" height="56" rx="6" fill="#0f172a" stroke="#d97706" stroke-width="1"/>
    <text x="90" y="78" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="monospace">W_unembed · z_K</text>
    <text x="90" y="98" text-anchor="middle" fill="#34d399" font-size="11" font-weight="700">Answer: "#### 111"</text>
  </g>

  <!-- Tap Line Going Down to RingBufferLatentTee -->
  <line x1="480" y1="235" x2="480" y2="280" stroke="#818cf8" stroke-width="2.5" stroke-dasharray="4,4" marker-end="url(#arrow-indigo)"/>
  <rect x="425" y="246" width="110" height="20" rx="4" fill="#312e81"/>
  <text x="480" y="260" text-anchor="middle" fill="#c7d2fe" font-size="10" font-weight="700">.detach() TAP</text>

  <!-- Phase 4: Detachable RingBufferLatentTee & SidecarProbe -->
  <g transform="translate(180, 285)">
    <rect width="600" height="165" rx="12" fill="url(#probeGrad)" stroke="#6366f1" stroke-width="2"/>
    <rect x="20" y="12" width="230" height="22" rx="4" fill="#3730a3"/>
    <text x="135" y="27" text-anchor="middle" fill="#e0e7ff" font-size="11" font-weight="700">DETACHABLE AUDITABILITY TEE</text>
    <text x="445" y="27" text-anchor="middle" fill="#a5b4fc" font-size="11">Bounded O(1) Memory (M ≤ K) • Default OFF in Prod</text>

    <!-- Ring Buffer Box -->
    <g transform="translate(30, 48)">
      <rect width="250" height="98" rx="8" fill="#1e293b" stroke="#4f46e5" stroke-width="1.5"/>
      <text x="125" y="24" text-anchor="middle" fill="#c7d2fe" font-size="12" font-weight="600">RingBufferLatentTee (M ≤ K)</text>
      <text x="125" y="44" text-anchor="middle" fill="#94a3b8" font-size="10">Circular Buffer: Memory O(M·d) = O(1)</text>
      <rect x="25" y="56" width="200" height="30" rx="4" fill="#0f172a" stroke="#4338ca" stroke-width="1"/>
      <text x="125" y="75" text-anchor="middle" fill="#818cf8" font-size="11" font-family="monospace">z̃_k = sg[z_k]  (No Grad)</text>
    </g>

    <!-- Sidecar Probes Box -->
    <g transform="translate(315, 48)">
      <rect width="255" height="98" rx="8" fill="#1e293b" stroke="#4f46e5" stroke-width="1.5"/>
      <text x="127" y="24" text-anchor="middle" fill="#c7d2fe" font-size="12" font-weight="600">Isolated Readout Heads</text>
      <text x="127" y="44" text-anchor="middle" fill="#f87171" font-size="10" font-weight="600">∇_θ_backbone ℒ_probe = 0 (Zero Contamination)</text>
      <text x="127" y="68" text-anchor="middle" fill="#e0e7ff" font-size="10">1. LogitLens(z̃_k) → Superposition Tokens</text>
      <text x="127" y="86" text-anchor="middle" fill="#e0e7ff" font-size="10">2. SidecarProbe g_φ(z̃_k) → Semantic Trace</text>
    </g>
  </g>
</svg>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Generated {out_path}")

def generate_frontier_svg(out_path: str):
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 460" width="100%" height="100%" style="background:#0f172a; border-radius:12px; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <!-- Title & Subtitle -->
  <text x="480" y="36" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="700">REASONING ACCURACY VS. COMPUTATIONAL COST (GSM8K)</text>
  <text x="480" y="56" text-anchor="middle" fill="#94a3b8" font-size="12">Continuous Latent Looping shifts the Pareto Frontier upward (+1.5% to +3.6% accuracy, -75% compute)</text>

  <!-- Plot Area: x from 120 to 880 (width 760), y from 90 to 390 (height 300) -->
  <!-- Grid Lines -->
  <line x1="120" y1="390" x2="880" y2="390" stroke="#334155" stroke-width="1.5"/>
  <line x1="120" y1="315" x2="880" y2="315" stroke="#1e293b" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="120" y1="240" x2="880" y2="240" stroke="#1e293b" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="120" y1="165" x2="880" y2="165" stroke="#1e293b" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="120" y1="90" x2="880" y2="90" stroke="#1e293b" stroke-width="1" stroke-dasharray="4,4"/>

  <line x1="120" y1="90" x2="120" y2="390" stroke="#334155" stroke-width="1.5"/>

  <!-- Y-Axis Labels (Accuracy %) -->
  <text x="105" y="394" text-anchor="end" fill="#94a3b8" font-size="11">0%</text>
  <text x="105" y="319" text-anchor="end" fill="#94a3b8" font-size="11">25%</text>
  <text x="105" y="244" text-anchor="end" fill="#94a3b8" font-size="11">50%</text>
  <text x="105" y="169" text-anchor="end" fill="#94a3b8" font-size="11">75%</text>
  <text x="105" y="94" text-anchor="end" fill="#94a3b8" font-size="11">100%</text>

  <!-- X-Axis Labels (Tokens & GFLOPs) -->
  <text x="210" y="415" text-anchor="middle" fill="#94a3b8" font-size="11">~5 Tokens</text>
  <text x="210" y="430" text-anchor="middle" fill="#64748b" font-size="10">(80 GFLOPs)</text>

  <text x="460" y="415" text-anchor="middle" fill="#94a3b8" font-size="11">~53 Tokens</text>
  <text x="460" y="430" text-anchor="middle" fill="#64748b" font-size="10">(130 GFLOPs)</text>

  <text x="780" y="415" text-anchor="middle" fill="#94a3b8" font-size="11">~125 Tokens</text>
  <text x="780" y="430" text-anchor="middle" fill="#64748b" font-size="10">(220 GFLOPs)</text>

  <text x="500" y="450" text-anchor="middle" fill="#cbd5e1" font-size="12" font-weight="600">Inference Compute (Mean Tokens Emitted &amp; GFLOPs)</text>

  <!-- Pareto Frontier Area Curve (Shaded) -->
  <path d="M 210 265 Q 460 215 780 152 L 780 390 L 210 390 Z" fill="#10b981" fill-opacity="0.08"/>
  <path d="M 210 265 Q 460 215 780 152" fill="none" stroke="#10b981" stroke-width="2.5" stroke-dasharray="6,4"/>
  <text x="350" y="210" fill="#10b981" font-size="11" font-weight="700">★ Global Pareto Frontier</text>

  <!-- Point 1: Verbose CoT (Slate) -->
  <circle cx="780" cy="152" r="8" fill="#64748b" stroke="#f8fafc" stroke-width="2"/>
  <rect x="680" y="110" width="190" height="34" rx="4" fill="#1e293b" stroke="#64748b" stroke-width="1"/>
  <text x="775" y="125" text-anchor="middle" fill="#f8fafc" font-size="11" font-weight="600">Verbose CoT Baseline</text>
  <text x="775" y="138" text-anchor="middle" fill="#94a3b8" font-size="10">79.2% Acc | 125 tok | 219 GF</text>

  <!-- Point 2: Telegraphic CoT (Amber) -->
  <circle cx="460" cy="227" r="8" fill="#f59e0b" stroke="#f8fafc" stroke-width="2"/>
  <rect x="370" y="245" width="180" height="34" rx="4" fill="#1e293b" stroke="#f59e0b" stroke-width="1"/>
  <text x="460" y="260" text-anchor="middle" fill="#fbbf24" font-size="11" font-weight="600">Telegraphic CoT Control</text>
  <text x="460" y="273" text-anchor="middle" fill="#94a3b8" font-size="10">54.2% Acc | 53 tok | -58% Tokens</text>

  <!-- Point 3: Compute-Matched (Red) - Collapsed -->
  <circle cx="470" cy="390" r="8" fill="#ef4444" stroke="#f8fafc" stroke-width="2"/>
  <rect x="485" y="350" width="180" height="34" rx="4" fill="#1e293b" stroke="#ef4444" stroke-width="1"/>
  <text x="575" y="365" text-anchor="middle" fill="#f87171" font-size="11" font-weight="600">Compute-Matched Truncated</text>
  <text x="575" y="378" text-anchor="middle" fill="#94a3b8" font-size="10">0.0% Acc (Grammar Starvation)</text>

  <!-- Latent Loop Points (Emerald) -->
  <!-- K=4 -->
  <circle cx="205" cy="390" r="5" fill="#10b981"/>
  <!-- K=8 -->
  <circle cx="210" cy="377" r="6" fill="#10b981"/>
  <!-- K=16 (Emerald Big Star) -->
  <circle cx="215" cy="265" r="9" fill="#10b981" stroke="#f8fafc" stroke-width="2"/>
  <rect x="130" y="195" width="185" height="46" rx="4" fill="#064e3b" stroke="#10b981" stroke-width="1.5"/>
  <text x="222" y="212" text-anchor="middle" fill="#a7f3d0" font-size="11" font-weight="700">Latent Loop (K=16)</text>
  <text x="222" y="226" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="600">41.7% Acc | Only 4.6 Tokens!</text>
  <text x="222" y="238" text-anchor="middle" fill="#6ee7b7" font-size="10">-60.5% FLOPs | -52.5% KV Cache</text>

  <!-- Legend Box -->
  <g transform="translate(620, 60)">
    <rect width="240" height="24" rx="6" fill="#1e293b" stroke="#334155" stroke-width="1"/>
    <circle cx="16" cy="12" r="5" fill="#64748b"/>
    <text x="26" y="16" fill="#cbd5e1" font-size="10">Verbose</text>
    <circle cx="80" cy="12" r="5" fill="#f59e0b"/>
    <text x="90" y="16" fill="#cbd5e1" font-size="10">Telegraphic</text>
    <circle cx="155" cy="12" r="5" fill="#ef4444"/>
    <text x="165" y="16" fill="#cbd5e1" font-size="10">Matched</text>
    <circle cx="210" cy="12" r="5" fill="#10b981"/>
    <text x="220" y="16" fill="#cbd5e1" font-size="10">Loop</text>
  </g>
</svg>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_architecture_svg("assets/gitltyn_architecture.svg")
    generate_frontier_svg("assets/pareto_frontier.svg")
