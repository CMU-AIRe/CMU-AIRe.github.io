/* Live gridworld for the AIRe homepage.
   An agent learns a maze in about 30 seconds, then moves on to a new one.

   The algorithm is plain one-step Q-learning and nothing else:
     - every step costs 1, and reaching the goal ends the run;
     - so a cell's value is minus the number of steps the agent thinks it
       still needs, which is the number drawn in the cell;
     - the agent always takes the move with the best value. Values start at
       zero, so a move it has never tried looks better than any move it has,
       and that alone makes it explore.

   The core (world + learner) is DOM-free so it can be tested in Node. */

const FIRST_MAZE = [
  ".....G",
  "...##.",
  ".#....",
  "#####.",
  "S.....",
];

const DX = [0, 1, 0, -1]; // up, right, down, left
const DY = [-1, 0, 1, 0];

/* Small seeded generator, so a run can be rehearsed and then replayed. */
function seeded(a) {
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeWorld(rows) {
  const H = rows.length, W = rows[0].length;
  const walls = new Uint8Array(W * H);
  let start = 0, goal = 0;
  rows.forEach((row, y) => [...row].forEach((ch, x) => {
    const i = y * W + x;
    if (ch === "#") walls[i] = 1;
    if (ch === "S") start = i;
    if (ch === "G") goal = i;
  }));
  return { W, H, walls, start, goal };
}

/* The cell reached by taking action `a` from `s`, or `s` itself if blocked. */
function move(world, s, a) {
  const x = (s % world.W) + DX[a], y = Math.floor(s / world.W) + DY[a];
  if (x < 0 || y < 0 || x >= world.W || y >= world.H) return s;
  const n = y * world.W + x;
  return world.walls[n] ? s : n;
}

/* Steps from the start to every cell; -1 where a cell cannot be reached. */
function distances(world) {
  const dist = new Int32Array(world.W * world.H).fill(-1);
  const queue = [world.start];
  dist[world.start] = 0;
  for (let q = 0; q < queue.length; q++) {
    const s = queue[q];
    for (let a = 0; a < 4; a++) {
      const n = move(world, s, a);
      if (dist[n] < 0) { dist[n] = dist[s] + 1; queue.push(n); }
    }
  }
  return dist;
}

const shortestRoute = (world) => distances(world)[world.goal];

/* A random maze built from short wall segments, with the start on the left
   edge and the goal on the right edge. */
function randomWorld(rand, W = 6, H = 5) {
  const pick = (list) => list[Math.floor(rand() * list.length)];
  for (let attempt = 0; attempt < 400; attempt++) {
    const walls = new Uint8Array(W * H);
    const bars = 3 + Math.floor(rand() * 3);
    for (let b = 0; b < bars; b++) {
      const across = rand() < 0.5, length = 2 + Math.floor(rand() * 2);
      const x = Math.floor(rand() * W), y = Math.floor(rand() * H);
      for (let k = 0; k < length; k++) {
        const cx = x + (across ? k : 0), cy = y + (across ? 0 : k);
        if (cx < W && cy < H) walls[cy * W + cx] = 1;
      }
    }
    const free = (test) => [...walls.keys()].filter((i) => !walls[i] && test(i % W));
    const left = free((x) => x === 0), right = free((x) => x === W - 1);
    if (!left.length || !right.length) continue;
    const world = { W, H, walls, start: pick(left), goal: pick(right) };
    const dist = distances(world);
    if (dist[world.goal] < 7 || dist[world.goal] > 11) continue;
    dist.forEach((d, i) => { if (d < 0) walls[i] = 1; }); // close off unreachable pockets
    const count = walls.reduce((sum, w) => sum + w, 0);
    if (count >= 5 && count <= 10) return world;
  }
  return makeWorld(FIRST_MAZE);
}

/* One-step Q-learning with a cost of 1 per step. See the note at the top. */
class Learner {
  constructor(world, rand = Math.random) {
    this.world = world;
    this.rand = rand;
    const n = world.W * world.H;
    this.Q = new Float64Array(n * 4);   // all zero: nothing is known yet
    this.tried = new Uint8Array(n * 4);
    // The agent can see which neighbouring cells are open, so it never walks into a wall.
    this.open = Array.from({ length: n }, (_, s) => [0, 1, 2, 3].filter((a) => move(world, s, a) !== s));
    this.episode = 0;
    this.lastSteps = 0;
    this.quiet = 0; // episodes in a row in which no value changed
    this.done = false;
    this.beginEpisode();
  }

  beginEpisode() {
    this.episode += 1;
    this.s = this.world.start;
    this.steps = 0;
    this.changed = 0;
    this.path = [this.s];
  }

  value(s) {
    if (s === this.world.goal) return 0;
    let v = -Infinity;
    for (const a of this.open[s]) v = Math.max(v, this.Q[s * 4 + a]);
    return v;
  }

  /* The agent's current estimate of the steps from `s` to the goal. */
  estimate(s) { return -this.value(s); }

  /* Best move; ties are broken at random unless `fixed` is set. */
  best(s, fixed = false) {
    const v = this.value(s);
    const ties = this.open[s].filter((a) => this.Q[s * 4 + a] === v);
    return fixed ? ties[0] : ties[Math.floor(this.rand() * ties.length)];
  }

  /* True once the best move from `s` is one the agent has actually tried. */
  knowsBest(s) { return this.tried[s * 4 + this.best(s, true)] === 1; }

  /* One step, and the one-step Q-learning update for it.
     Returns true when the run has ended. */
  step(fixed = false) {
    const s = this.s, a = this.best(s, fixed);
    const n = move(this.world, s, a);
    const target = -1 + this.value(n);
    if (this.Q[s * 4 + a] !== target) { this.Q[s * 4 + a] = target; this.changed += 1; }
    this.tried[s * 4 + a] = 1;
    this.s = n;
    this.steps += 1;
    this.path.push(n);
    return n === this.world.goal || this.steps >= 3000;
  }

  endEpisode() {
    this.lastSteps = this.steps;
    this.quiet = this.changed === 0 ? this.quiet + 1 : 0;
    // Two runs with nothing left to learn: the values have settled.
    if (this.quiet >= 2 || this.episode >= 80) this.done = true;
  }
}

/* Run a whole learning session ahead of time and report how long each
   run takes, so the live session can be paced to a fixed duration. */
function rehearse(world, seed) {
  const learner = new Learner(world, seeded(seed));
  const lengths = [];
  while (!learner.done) {
    while (!learner.step()) { /* run to the goal */ }
    lengths.push(learner.steps);
    learner.endEpisode();
    if (!learner.done) learner.beginEpisode();
  }
  const total = lengths.reduce((sum, n) => sum + n, 0);
  return { lengths, total, route: learner.lastSteps };
}

/* Pick a seed whose session is neither over in a flash nor too long to play calmly. */
function chooseRun(world, rand, tries = 40) {
  let fallback = null;
  for (let attempt = 0; attempt < tries; attempt++) {
    const seed = Math.floor(rand() * 2 ** 31);
    const run = { seed, ...rehearse(world, seed) };
    fallback = fallback || run;
    if (run.total >= 170 && run.total <= 330) return { ...run, fits: true };
  }
  return { ...fallback, fits: false };
}

if (typeof module !== "undefined") {
  module.exports = { FIRST_MAZE, makeWorld, randomWorld, shortestRoute, Learner, rehearse, chooseRun, seeded };
}

/* ---------- View ---------- */

if (typeof document !== "undefined") (function () {
  const host = document.querySelector("[data-gridworld]");
  if (!host) return;
  const svg = host.querySelector("svg");
  const caption = host.querySelector("[data-caption]");
  const nextButton = host.querySelector("[data-next]");
  const NS = "http://www.w3.org/2000/svg";
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Pacing, in milliseconds. One maze takes about 30 seconds in total. */
  const LEARN_MS = 22000; // all the learning steps together
  const SPEED_UP = 2.2;   // the last learning step is this many times faster than the first
  const WALK_MS = 110;    // per step, when following the learned route
  const HOLD_MS = 3000;   // pause on the finished maze
  const FADE_MS = 450;    // fade out before the next maze fades in

  /* Colours: walls are dark tiles, open cells are pale ones, and an open tile
     takes on more of the accent the closer the agent thinks it is to the goal. */
  const WALL = "#22252b";
  const TILE = [240, 241, 244], ACCENT = [63, 98, 173];
  const blend = (t) => `rgb(${TILE.map((c, k) => Math.round(c + (ACCENT[k] - c) * t)).join(",")})`;

  let world = makeWorld(FIRST_MAZE);
  let learner, run, phase, acc = 0, hold = 0, restart = false;
  let upcoming = null;                            // the maze waiting behind a fade
  let ax = 0, ay = 0, alpha = 0, snap = true;     // where the agent is drawn, eased toward its cell
  const { W, H } = world;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const el = (name, attrs) => {
    const node = document.createElementNS(NS, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    svg.appendChild(node);
    return node;
  };

  const GAP = 0.08;
  const cells = [], arrows = [];
  for (let i = 0; i < W * H; i++) {
    const x = (i % W) + GAP / 2, y = Math.floor(i / W) + GAP / 2;
    const rect = el("rect", { x, y, width: 1 - GAP, height: 1 - GAP, rx: 0.09, class: "gw-cell" });
    rect.addEventListener("click", () => toggleWall(i));
    cells.push(rect);
  }
  for (let i = 0; i < W * H; i++) arrows.push(el("path", { d: "M-.075 .04 L0 -.04 L.075 .04", class: "gw-arrow" }));
  const startRing = el("circle", { r: 0.13, class: "gw-start" });
  const trail = el("polyline", { class: "gw-trail" });
  const agent = el("circle", { r: 0.15, class: "gw-agent" });

  const cx = (s) => (s % W) + 0.5, cy = (s) => Math.floor(s / W) + 0.5;

  function draw(dt = 16) {
    for (let i = 0; i < W * H; i++) {
      const cell = cells[i], arrow = arrows[i];
      arrow.style.opacity = 0;
      if (world.walls[i]) { cell.style.fill = WALL; continue; }
      if (i === world.goal) { cell.style.fill = blend(1); continue; }
      // A tile is tinted only once the agent knows its best move from there.
      if (!learner.knowsBest(i)) { cell.style.fill = blend(0); continue; }
      cell.style.fill = blend(0.07 + 0.6 * Math.pow(0.8, learner.estimate(i) - 1)); // fewer steps left = deeper
      arrow.style.opacity = 1;
      arrow.setAttribute("transform", `translate(${cx(i)} ${cy(i)}) rotate(${learner.best(i, true) * 90})`);
    }
    startRing.setAttribute("cx", cx(world.start));
    startRing.setAttribute("cy", cy(world.start));
    // The agent glides toward its cell rather than jumping, and fades in after a reset.
    const tx = cx(learner.s), ty = cy(learner.s);
    if (snap) { ax = tx; ay = ty; snap = false; }
    const ease = 1 - Math.exp(-dt / (0.5 * stepDuration()));
    ax += (tx - ax) * ease;
    ay += (ty - ay) * ease;
    alpha += (1 - alpha) * (1 - Math.exp(-dt / 160));
    agent.setAttribute("cx", ax);
    agent.setAttribute("cy", ay);
    agent.style.opacity = alpha;
    // No trail while learning. At the end the learned route is drawn behind the agent.
    const route = phase === "learn" ? [] : learner.path.slice(0, -1).map((p) => `${cx(p)},${cy(p)}`).concat(`${ax},${ay}`);
    trail.setAttribute("points", route.join(" "));
  }

  function describe() {
    if (!caption) return; // the caption is optional
    if (phase === "walk" || phase === "hold") {
      const best = shortestRoute(world), found = learner.lastSteps;
      caption.textContent = found === best
        ? `Shortest route found in ${learner.episode} runs: ${found} steps.`
        : `Settled on ${found} steps after ${learner.episode} runs. The shortest route is ${best}.`;
    } else if (learner.episode === 1) {
      caption.textContent = "Q-learning, run 1. Nothing is known yet.";
    } else {
      caption.textContent = `Q-learning, run ${learner.episode}. The last run took ${learner.lastSteps} steps.`;
    }
  }

  /* Each learning step is a little shorter than the one before. */
  let firstStep = 0, shrink = 1, taken = 0;
  function setPace() {
    shrink = Math.pow(1 / SPEED_UP, 1 / Math.max(1, run.total - 1));
    firstStep = Math.min(190, (LEARN_MS * (1 - shrink)) / (1 - Math.pow(shrink, run.total)));
    taken = 0;
  }
  function stepDuration() {
    return phase === "walk" ? WALK_MS : firstStep * Math.pow(shrink, taken);
  }

  /* Advance by one tick. Returns true when a run has just ended. */
  function advance() {
    if (restart) { // the agent rested on the goal for a tick; now put it back at the start
      restart = false;
      learner.beginEpisode();
      snap = true;
      alpha = 0;
      describe();
      return false;
    }
    if (phase === "learn") taken += 1;
    if (!learner.step(phase === "walk")) return false;
    if (phase === "walk") {
      phase = "hold";
      hold = HOLD_MS;
    } else {
      learner.endEpisode();
      if (learner.done) phase = "walk";
      restart = true;
    }
    describe();
    return true;
  }

  function startMaze(next) {
    world = next;
    run = chooseRun(world, Math.random);
    learner = new Learner(world, seeded(run.seed));
    phase = "learn";
    acc = 0;
    restart = false;
    snap = true;
    alpha = still ? 1 : 0;
    setPace();
    describe();
    if (still) { while (phase !== "hold") advance(); snap = true; }
    draw();
    svg.style.opacity = 1;
    wake();
  }

  /* Fade the current maze out, then bring the next one in. */
  function changeMaze(next) {
    if (still) { startMaze(next); return; }
    upcoming = next;
    phase = "fade";
    hold = FADE_MS;
    svg.style.opacity = 0;
    wake();
  }

  /* A fresh random maze whose learning session fits the time budget. */
  function freshMaze() {
    let candidate = randomWorld(Math.random);
    for (let k = 0; k < 20 && !chooseRun(candidate, Math.random, 4).fits; k++) candidate = randomWorld(Math.random);
    return candidate;
  }

  let visible = true, previous = 0, raf = 0;
  function frame(now) {
    raf = 0;
    if (!visible || document.hidden) { previous = 0; return; }
    const dt = previous ? Math.min(now - previous, 100) : 16;
    previous = now;
    if (phase === "fade") {
      hold -= dt;
      if (hold <= 0) { startMaze(upcoming); return; }
    } else if (phase === "hold") {
      hold -= dt;
      if (hold <= 0) changeMaze(freshMaze());
    } else {
      acc += dt;
      for (let n = 0; n < 60 && (phase === "learn" || phase === "walk"); n++) {
        const ms = stepDuration();
        if (acc < ms) break;
        acc -= ms;
        if (advance()) { acc = -1.5 * ms; break; } // rest on the goal for a moment
      }
    }
    draw(dt);
    raf = requestAnimationFrame(frame);
  }
  function wake() { if (!raf && !still) raf = requestAnimationFrame(frame); }

  function toggleWall(i) {
    if (i === world.start || i === world.goal) return;
    world.walls[i] ^= 1;
    if (shortestRoute(world) < 0) { world.walls[i] ^= 1; return; } // keep the goal reachable
    startMaze(world);
  }

  if (nextButton) nextButton.addEventListener("click", () => changeMaze(freshMaze()));
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; if (visible) wake(); }).observe(svg);
  document.addEventListener("visibilitychange", wake);

  startMaze(world);
})();
