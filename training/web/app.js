"use strict";
const $ = id => document.getElementById(id);
let mode = "ours", data = null, sequence = null, playing = false, index = 0;
let yaw = -0.8, pitch = 0.3, zoom = 1, center = [0, 0, 0.8], span = 2;
const source = "https://github.com/nvidia-isaac/video_to_data/blob/8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef/robotic_grounding/flash_chord/src/flash_chord/";
const local = "https://github.com/Gelrignard/scalable_v2d/blob/work/tier2-rl/training/";
const details = {
  reference: ["The reference is a command, not an action demonstration.", "Body motion enters SONIC's encoder. Finger reference targets bypass SONIC. Object poses and contact regions condition the residual actor and task reward. Our original/corrected comparison changes only the chosen same-tier reference.", local + "runtime.py", "runtime.py · _make_goals"],
  encoder: ["SONIC's encoder stays frozen.", "The pinned runtime encodes the current body reference plus nine future samples, ending at +0.9 s. Our adapter also caches these 64-D latents as goal features for learning; the native action path uses the same frozen encoder.", source + "runtime/actions/sonic.py#L228", "sonic.py · build_sonic_encoder_reference"],
  decoder: ["This is the body feedback controller.", "The 64-D motion latent joins a 930-D history: ten frames of body angular velocity, joint positions, joint velocities, previous SONIC actions and projected gravity. The frozen decoder produces 29 body outputs. Object contacts influence it through the resulting body motion, rather than a direct object-goal input.", source + "runtime/actions/sonic.py", "sonic.py · SonicJointResidualAction"],
  actor: ["A task policy changes how the robot manipulates.", "Our actor outputs 43 normalized values: 29 body slots plus 14 finger slots. Hip, knee and ankle residuals are masked to zero in this pilot, leaving 31 effective learned controls. Live robot–object contacts are included in our actor observations. NVIDIA's whole-body observation path excludes explicit contact features; contacts still enter its reward.", local + "runtime.py", "runtime.py · action_mask / actor_observation"],
  targets: ["The residual is added after SONIC decoding.", "Our normalized action is multiplied by 3. Body targets = default + joint scale × (SONIC output + 0.15 × residual). Finger targets = reference + reset offset + 0.15 × residual. Position drives convert these targets into actuator forces in simulation. Residuals do not edit SONIC's weights or latent.", source + "runtime/actions/sonic.py#L367", "sonic.py · process_sonic_joint_targets"],
  physics: ["The object has to move through simulated contact.", "The pinned runtime provides Newton/MuJoCo-Warp physics, G1/Dex3 assets, actuator limits, contacts and target delays. Our configuration uses 50 Hz control and four physics substeps. Virtual object assistance is zero. During final evaluation there are no phase resets or object teleports.", local + "runtime.py", "runtime.py · SonicEnv"],
  feedback: ["Both action generators close the loop.", "After each physics step, actual robot motion refreshes SONIC's feedback history. Robot/object observations and goals feed the next residual action. Actual contacts support the wrench reward. The reference proposes useful motion; the simulator provides the dynamics evidence.", local + "runtime.py", "runtime.py · state / step"],
  learning: ["The world model is a training accelerator hypothesis.", "Fit dynamics from executed state/action/next-state, reward and termination records, including failures. Continue SAC from real data. Only enable short imagined actor updates after held-out model validation. Gradients pass through the learned model, not through frozen ONNX SONIC or contact physics. Reduced training cost is the experiment's goal; it is not yet measured.", local + "learner.py", "learner.py · model and actor updates"]
};
function showDetail(key) {
  const d = details[key];
  if (key === "learning" && mode === "chord") {
    $("detail").innerHTML = `<b>NVIDIA's pinned baseline trains with FlashSAC.</b><br>Real simulator transitions fill replay; actor and critic learn a per-sequence residual policy using object tracking, body/palm tracking, contact-wrench and regularization rewards. The baseline does not use our small learned dynamics model. <a href="${source}configs/experiment/g1_recon_body_flash_sac.yaml" target="_blank" rel="noopener">Pinned recipe ↗</a>`;
  } else $("detail").innerHTML = `<b>${d[0]}</b><br>${d[1]} <a href="${d[2]}" target="_blank" rel="noopener">${d[3]} ↗</a>`;
}
document.querySelectorAll(".node").forEach(node => {
  node.tabIndex = 0; node.setAttribute("role", "button");
  node.onclick = () => showDetail(node.dataset.detail);
  node.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); node.click(); } };
});
document.querySelectorAll("nav button").forEach(button => button.onclick = () => {
  document.querySelectorAll("nav button,.panel").forEach(n => n.classList.remove("active"));
  button.classList.add("active"); $(button.dataset.panel).classList.add("active"); draw();
});
function setMode(value) {
  mode = value;
  $("ours").classList.toggle("active", mode === "ours"); $("chord").classList.toggle("active", mode === "chord");
  $("mode-note").textContent = mode === "ours" ? "Our corrected motion is a soft reference. Live contacts and object motion decide whether actions work." : "NVIDIA's whole-body CHORD also learns a residual on a pretrained SONIC controller. This view follows the pinned FlashCHORD G1 runtime.";
  $("reference-text").textContent = mode === "ours" ? "Original or corrected robot motion, object goals, contact regions." : "Retargeted whole-body motion, object goals and human contact-wrench guidance.";
  $("actor-text").textContent = mode === "ours" ? "Robot state + object goals + live contacts → arm, waist & finger corrections" : "Robot state + reference commands → body & finger residuals; explicit contacts omitted from whole-body actor input";
  $("learner-tag").textContent = mode === "ours" ? "SAC + LEARNED DYNAMICS" : "FLASHSAC · MODEL-FREE RL";
  $("learner-text").textContent = mode === "ours" ? "Real-transition SAC. Fit a small world model; enable short imagined actor updates after validation." : "Actor/critic updates from replayed simulator transitions. Contact-wrench guidance helps discover effective manipulation.";
  showDetail("learning");
}
$("ours").onclick = () => setMode("ours"); $("chord").onclick = () => setMode("chord");
function mix() {
  const a = Number($("action").value), joint = $("joint").value;
  const executed = joint === "leg" ? 0 : a, residual = 3 * executed;
  const target = joint === "finger" ? 0.4 + 0.15 * residual : 0.25 * (0.8 + 0.15 * residual);
  $("action-value").textContent = a.toFixed(2);
  $("formula").innerHTML = `Our pilot: raw residual = 3 × ${executed.toFixed(2)} = ${residual.toFixed(2)}<br>` +
    (joint === "finger" ? `q target = 0.4 + 0.15 × ${residual.toFixed(2)}` : `q target = 0 + 0.25 × (0.8 + 0.15 × ${residual.toFixed(2)})`) +
    `<br><b>Joint target = ${target.toFixed(4)} rad</b>` + (joint === "leg" ? "<br>SONIC still moves this joint; only our added residual is zero." : "");
}
$("action").oninput = mix; $("joint").onchange = mix; mix();
function budget() {
  const h = Number($("hours").value), p = Number($("price4090").value), q = Number($("price3090").value);
  if (!(h > 0 && p > 0 && q > 0)) { $("budget").textContent = "Enter positive hours and prices."; return; }
  $("budget").innerHTML = `8 runs × ${h} h = ${(8*h).toFixed(1)} GPU-hours<br>1 × 4090: ≤ ${(8*h).toFixed(1)} training hours / ¥${(8*h*p).toFixed(2)}<br>2 × 3090 rental rate: ¥${(2*q).toFixed(2)} per wall hour<br><b>For equal work, the 3090 pair must be &gt; ${(2*q/p).toFixed(2)}× as fast to cost less.</b>`;
}
["hours", "price4090", "price3090"].forEach(id => $(id).oninput = budget); budget();
function choose() {
  if (!data) return;
  sequence = data.sequences.find(s => s.episode === Number($("episode").value) && s.tier === Number($("tier").value));
  index = 0; playing = false; $("play").textContent = "Play";
  $("frame").max = sequence.times.length - 1;
  const all = sequence.variants.original.points.flat().concat(sequence.variants.corrected.points.flat());
  const lo = [Infinity,Infinity,Infinity], hi = [-Infinity,-Infinity,-Infinity];
  all.forEach(p => p.forEach((v,k) => {lo[k] = Math.min(lo[k],v); hi[k] = Math.max(hi[k],v);}));
  center = lo.map((v,k) => (v+hi[k])/2); span = Math.max(...hi.map((v,k) => v-lo[k]), 1.5);
  yaw = -0.8; pitch = 0.3; zoom = 1; draw();
}
$("episode").onchange = choose; $("tier").onchange = choose;
$("frame").oninput = () => {index = Number($("frame").value); draw();};
$("play").onclick = () => {playing = !playing; $("play").textContent = playing ? "Pause" : "Play";};
["show-original", "show-corrected"].forEach(id => $(id).onchange = draw);
$("reset-camera").onclick = () => {yaw = -0.8; pitch = 0.3; zoom = 1; draw();};
const canvas = $("canvas"), ctx = canvas.getContext("2d");
let drag = null;
canvas.onpointerdown = e => {drag = [e.clientX,e.clientY]; canvas.setPointerCapture(e.pointerId);};
canvas.onpointerup = canvas.onpointercancel = () => {drag = null;};
canvas.onpointermove = e => {if (drag) {yaw += (e.clientX-drag[0])*.008; pitch = Math.max(-1.1,Math.min(1.1,pitch+(e.clientY-drag[1])*.008)); drag = [e.clientX,e.clientY]; draw();}};
canvas.addEventListener("wheel", e => {e.preventDefault(); zoom = Math.max(.35,Math.min(5,zoom*Math.exp(-e.deltaY*.001))); draw();}, {passive:false});
function draw() {
  if (!sequence || !$("references").classList.contains("active")) return;
  const width = canvas.clientWidth, height = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr); ctx.setTransform(dpr,0,0,dpr,0,0);
  const scale = Math.min(width,height)*.7/span*zoom;
  function project(point) {
    const x = point[0]-center[0], y = point[1]-center[1], z = point[2]-center[2];
    const side = x*Math.cos(yaw)-y*Math.sin(yaw), depth = x*Math.sin(yaw)+y*Math.cos(yaw);
    return [width/2+side*scale,height*.51-(z*Math.cos(pitch)-depth*Math.sin(pitch))*scale];
  }
  function line(a,b,color,w=1,dash=[]) {
    const p=project(a),q=project(b); ctx.strokeStyle=color; ctx.lineWidth=w; ctx.setLineDash(dash);
    ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();ctx.setLineDash([]);
  }
  const x = Math.floor(center[0]), y = Math.floor(center[1]);
  for (let k=-4;k<=4;k++) {line([x+k*.5,y-2,0],[x+k*.5,y+2,0],"#d0dcd9");line([x-2,y+k*.5,0],[x+2,y+k*.5,0],"#d0dcd9");}
  ["original","corrected"].forEach(v => {
    if (!$("show-"+v).checked) return;
    const s=sequence.variants[v], pts=s.points[index], color=v === "original" ? "#7b8798" : "#087a65";
    data.edges.forEach(([a,b]) => line(pts[a],pts[b],color,v === "original" ? 3 : 2));
    pts.forEach(p => {ctx.beginPath();ctx.fillStyle=color;ctx.arc(...project(p),2,0,Math.PI*2);ctx.fill();});
    const corners=s.object_boxes[index];
    for(let a=0;a<8;a++) for (const bit of [1,2,4]) if (!(a&bit)) line(corners[a],corners[a|bit],v === "original" ? "#bd8e69" : "#b35b18",1.3,[4,3]);
    for (let t=1;t<s.object_path.length;t++) line(s.object_path[t-1],s.object_path[t],"#cf9b7150",1);
  });
  $("frame").value=index; $("time").textContent=sequence.times[index].toFixed(2)+" s";
  $("duration").textContent=sequence.times.at(-1).toFixed(2)+" s";
  $("source-frame").textContent=sequence.frames[index]+" / "+(sequence.source_frames-1);
  $("delta").textContent=sequence.max_link_delta_mm[index].toFixed(1)+" mm";
  $("observed").textContent=sequence.variants.corrected.observed[index] ? "Yes" : "Completed / missing";
}
new ResizeObserver(draw).observe(canvas);
let last = 0;
function tick(now) {
  if (playing && sequence && $("references").classList.contains("active") && now-last >= 100) {
    index = (index+1)%sequence.times.length; last=now; draw();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
fetch("data.json").then(r => {if(!r.ok) throw new Error("HTTP "+r.status); return r.json();}).then(value => {
  data=value; $("commit").textContent=data.runtime_commit.slice(0,12); choose();
}).catch(error => {$("time").textContent="Load failed"; $("detail").textContent="Reference data could not load: "+error.message;});
