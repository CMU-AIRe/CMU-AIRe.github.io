/* A swarm that learns to get through a maze.

   About two hundred agents start on one side of an obstacle course and try to
   reach a target on the other side. They share one table of values: for each
   cell, how many steps they believe it takes to reach the target from there.
   Each round every agent walks until it reaches the target or runs out of
   steps, and the table is updated from what they did. Within a few rounds the
   swarm goes from searching to flowing; then the target moves and the swarm,
   still believing in the old spot, has to find it again.

   The learning is temporal-difference learning of state values, shared by all
   agents (the agents are parallel explorers of one learner):
     - a cell's value V(s) is the number of steps still needed; the target is
       worth 0 and every step costs 1;
     - after a visit, V(s) moves halfway toward 1 + the value of its best
       neighbour (a one-step Bellman backup);
     - at the end of a round each walk is replayed backwards, so news of the
       target travels back along a whole walk in one round, not one cell;
     - a cell nobody has visited is assumed to be close to the target
       (optimism), which is what drives the first round's exploration.
   Agents see which neighbouring cells are open and choose among them by a
   softmax over the neighbours' values, with a slight preference for keeping
   their direction.

   Everything is rehearsed in memory ahead of time, then replayed on screen at
   a calm pace; among a few random courses and seeds, one whose first round
   mostly fails and that settles within a few rounds is chosen. The core
   (world + learner + planner) has no DOM code, so it runs in Node too. */

/* ---------- Core: the world ---------- */

/* Moves: four straight, then four diagonal. A diagonal move is allowed only
   when both straight cells beside it are open, so agents never cut a corner. */
const DX = [1, 0, -1, 0, 1, -1, -1, 1];
const DY = [0, 1, 0, -1, 1, 1, -1, -1];

/* Small seeded generator, so a session can be rehearsed and replayed exactly. */
function seeded(a) {
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* A world is a W x H grid of open cells and walls, a start cell and a goal cell.
   `next[s]` lists the cells reachable from s in one move, `dir[s]` the
   direction of each of those moves, and `startArea` the cells agents start on. */
function makeWorld(W, H, walls, start, goal) {
  const next = [], dir = [];
  const open = (x, y) => x >= 0 && y >= 0 && x < W && y < H && !walls[y * W + x];
  for (let s = 0; s < W * H; s++) {
    const x = s % W, y = (s - x) / W, cells = [], dirs = [];
    if (!walls[s]) {
      for (let d = 0; d < 8; d++) {
        const nx = x + DX[d], ny = y + DY[d];
        if (!open(nx, ny)) continue;
        if (d >= 4 && !(open(nx, y) && open(x, ny))) continue;
        cells.push(ny * W + nx);
        dirs.push(d);
      }
    }
    next.push(cells);
    dir.push(dirs);
  }
  const startArea = next[start].filter((s) => s !== goal).concat(start);
  return { W, H, walls, start, goal, next, dir, startArea };
}

/* Fewest moves from `from` to every cell; -1 where a cell cannot be reached. */
function distances(world, from) {
  const dist = new Int32Array(world.W * world.H).fill(-1);
  const queue = [from];
  dist[from] = 0;
  for (let q = 0; q < queue.length; q++) {
    const s = queue[q];
    for (const n of world.next[s]) if (dist[n] < 0) { dist[n] = dist[s] + 1; queue.push(n); }
  }
  return dist;
}

const shortestRoute = (world) => distances(world, world.start)[world.goal];

/* A random obstacle course laid out from left to right on an L x C grid: a few
   walls across the field, each with an opening that does not line up with the
   previous one, and some short loose walls in between. The route must wind a
   little but not too much; on a grid too small for that, any route will do. */
function courseLeftToRight(rand, L, C) {
  const int = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  for (let attempt = 0; ; attempt++) {
    const strict = attempt < 200;
    const walls = new Uint8Array(L * C);
    const set = (x, y) => { if (x >= 0 && y >= 0 && x < L && y < C) walls[y * L + x] = 1; };
    const barriers = Math.max(2, Math.min(5, Math.round((L - 6) / 6)));
    let lastGap = -1;
    for (let b = 0; b < barriers; b++) {
      const x = Math.round(3 + ((L - 7) * (b + 0.5)) / barriers) + int(-1, 1);
      const gap = Math.max(2, Math.round(C / 6)) + int(0, 1);
      let top = int(0, C - gap);
      for (let k = 0; k < 20 && lastGap >= 0 && Math.abs(top + gap / 2 - lastGap) < C / 3; k++) top = int(0, C - gap);
      lastGap = top + gap / 2;
      for (let y = 0; y < C; y++) if (y < top || y >= top + gap) set(x, y);
    }
    // Loose walls stand free: none touches another wall or the edge.
    const clear = (x, y) => {
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && ny >= 0 && nx < L && ny < C && walls[ny * L + nx]) return false;
      }
      return true;
    };
    const loose = Math.round((L * C) / 70);
    for (let k = 0; k < loose; k++) {
      const across = rand() < 0.5, length = int(2, 3), x = int(2, L - 3), y = int(1, C - 2);
      const cells = Array.from({ length }, (_, i) => [x + (across ? i : 0), y + (across ? 0 : i)]);
      if (cells.every(([cx, cy]) => cx < L - 1 && cy < C - 1 && clear(cx, cy))) for (const [cx, cy] of cells) set(cx, cy);
    }
    const start = int(1, C - 2) * L + 1;
    const goal = int(1, C - 2) * L + (L - 2);
    // Keep the start, the goal and the cells around them open.
    for (const c of [start, goal]) {
      const cx = c % L, cy = (c - cx) / L;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const x = cx + dx, y = cy + dy;
        if (x >= 0 && y >= 0 && x < L && y < C) walls[y * L + x] = 0;
      }
    }
    const dist = distances(makeWorld(L, C, walls, start, goal), start), route = dist[goal];
    if (route < 0 || (strict && (route < L * 1.05 || route > L * 1.6))) continue;
    // Wall off pockets that cannot be reached, so every open cell is part of the course.
    dist.forEach((d, i) => { if (d < 0) walls[i] = 1; });
    return { walls, start, goal };
  }
}

/* A course for a W x H grid. The route runs along the longer side: left to
   right on a wide grid, bottom to top on a tall one; `flip` reverses it. */
function randomCourse(rand, W, H, flip = false) {
  const tall = H > W, L = tall ? H : W, C = tall ? W : H;
  const base = courseLeftToRight(rand, L, C);
  const place = (s) => {
    let x = s % L;
    const y = (s - x) / L;
    if (flip) x = L - 1 - x;
    return tall ? (L - 1 - x) * W + y : y * W + x;
  };
  const walls = new Uint8Array(W * H);
  base.walls.forEach((w, s) => { if (w) walls[place(s)] = 1; });
  return makeWorld(W, H, walls, place(base.start), place(base.goal));
}

/* The same walls with the goal moved to `goal`, or null if that cell is a
   wall, cannot be reached, or is too close to the start to be interesting. */
function withGoal(world, goal) {
  if (world.walls[goal] || goal === world.start) return null;
  if (distances(world, world.start)[goal] < 4) return null;
  return makeWorld(world.W, world.H, world.walls, world.start, goal);
}

/* ---------- Core: the learner ---------- */

const SETTINGS = {
  rate: 0.5,        // how far a value moves toward its new target in one update
  temperature: 0.5, // softmax temperature, in steps: lower is greedier
  hope: 10,         // the value of a cell nobody has visited yet (optimism)
  keep: 1.0,        // preference for keeping the same direction
  patience: 1.7,    // a round lasts this many times the shortest route (+6 steps)
};

/* One table of values shared by every agent. `values`, if given, is what the
   swarm already believes (from an earlier goal); otherwise it starts fresh. */
class Swarm {
  constructor(world, rand, agents, values = null, settings = SETTINGS) {
    this.world = world;
    this.rand = rand;
    this.agents = agents;
    this.settings = settings;
    this.maxSteps = Math.round(settings.patience * shortestRoute(world)) + 6;
    this.V = values ? Float32Array.from(values) : new Float32Array(world.W * world.H).fill(settings.hope);
    this.V[world.goal] = 0;
  }

  /* Move V(s) halfway toward one step more than its best neighbour. */
  update(s) {
    if (s === this.world.goal) return;
    let best = Infinity;
    for (const n of this.world.next[s]) best = Math.min(best, this.V[n]);
    this.V[s] += this.settings.rate * (1 + best - this.V[s]);
  }

  /* Pick a move for an agent at `s` that last moved in direction `last`;
     returns its index in world.next[s]. */
  choose(s, last) {
    const { temperature, keep } = this.settings;
    const cells = this.world.next[s], dirs = this.world.dir[s];
    const score = new Float64Array(cells.length);
    let top = -Infinity;
    for (let i = 0; i < cells.length; i++) {
      score[i] = -this.V[cells[i]] / temperature + (dirs[i] === last ? keep : 0);
      top = Math.max(top, score[i]);
    }
    let total = 0;
    for (let i = 0; i < score.length; i++) total += (score[i] = Math.exp(score[i] - top));
    let r = this.rand() * total, i = 0;
    while (i < score.length - 1 && (r -= score[i]) > 0) i++;
    return i;
  }

  /* Play one round: every agent walks until it reaches the goal or runs out of
     steps. Returns the walks (lists of cells) and how many reached the goal. */
  playRound() {
    const { world, agents, maxSteps } = this;
    const walks = [], at = new Int32Array(agents), last = new Int8Array(agents).fill(-1);
    for (let a = 0; a < agents; a++) {
      at[a] = world.startArea[a % world.startArea.length];
      walks.push([at[a]]);
    }
    let walking = agents;
    for (let t = 0; t < maxSteps && walking > 0; t++) {
      for (let a = 0; a < agents; a++) {
        const s = at[a];
        if (s === world.goal) continue;
        const i = this.choose(s, last[a]), n = world.next[s][i];
        last[a] = world.dir[s][i];
        at[a] = n;
        walks[a].push(n);
        this.update(n);
        this.update(s);
        if (n === world.goal) walking -= 1;
      }
    }
    // Replay every walk backwards, so what was learned at its end reaches its start.
    for (const walk of walks) for (let t = walk.length - 1; t >= 0; t--) this.update(walk[t]);
    return { walks, reached: agents - walking };
  }
}

/* Play a learning session in memory, round after round, until it has settled
   (at least 95% arrive, on routes at most 25% longer than the shortest on
   average) and at least `least` rounds have been played, or until `limit`
   rounds. Each round keeps its walks and the values before and after it. */
function rehearse(world, seed, agents, { least = 3, limit = 6, values = null } = {}) {
  const swarm = new Swarm(world, seeded(seed), agents, values);
  const best = shortestRoute(world), rounds = [];
  let settled = false;
  while (rounds.length < least || (rounds.length < limit && !settled)) {
    const before = swarm.V.slice();
    const { walks, reached } = swarm.playRound();
    let total = 0, steps = 0;
    for (const w of walks) {
      steps = Math.max(steps, w.length - 1);
      if (w[w.length - 1] === world.goal) total += w.length - 1;
    }
    const meanSteps = reached ? total / reached : 0;
    rounds.push({ walks, before, after: swarm.V.slice(), reached, share: reached / agents, meanSteps, steps });
    settled = reached >= 0.95 * agents && meanSteps <= 1.25 * best;
  }
  return { world, agents, best, rounds, settled };
}

/* How well a session would play on screen: lower is better. The ideal first
   round has a few arrivals (between 2% and 40%), the second a clear majority,
   and the whole session settles in at most `rounds` rounds. */
function misfit(session, { firstLow = 0.02, firstHigh = 0.4, rounds = 4 } = {}) {
  const [first, second] = session.rounds;
  let cost = session.settled ? Math.max(0, session.rounds.length - rounds) : 10;
  if (first.share < firstLow) cost += 2 + (firstLow - first.share) * 50;
  if (first.share > firstHigh) cost += 2 + (first.share - firstHigh) * 10;
  if (second && second.share < 0.5) cost += 2;
  return cost;
}

/* A new course and a learning session on it that plays well. */
function planCourse(rand, W, H, agents, tries = 8) {
  let pick = null, cost = Infinity;
  for (let k = 0; k < tries && cost > 0; k++) {
    const world = randomCourse(rand, W, H, rand() < 0.5);
    const session = rehearse(world, Math.floor(rand() * 2 ** 31), agents);
    const c = misfit(session);
    if (c < cost) { pick = session; cost = c; }
  }
  return pick;
}

/* Fewest moves from the nearest of `cells` to every cell. */
function distancesFrom(world, cells) {
  const dist = new Int32Array(world.W * world.H).fill(-1);
  const queue = [...cells];
  for (const c of cells) dist[c] = 0;
  for (let q = 0; q < queue.length; q++) {
    const s = queue[q];
    for (const n of world.next[s]) if (dist[n] < 0) { dist[n] = dist[s] + 1; queue.push(n); }
  }
  return dist;
}

/* The route the swarm believes in: downhill on `values` from the start. */
function believedRoute(world, values) {
  const route = [world.start];
  for (let s = world.start; values[s] > 0.5 && route.length < world.W * world.H;) {
    let m = -1;
    for (const c of world.next[s]) if (m < 0 || values[c] < values[m]) m = c;
    if (m < 0 || values[m] >= values[s]) break;
    route.push((s = m));
  }
  return route;
}

/* The goal moves while the swarm keeps what it has learned (`values`). With a
   `goal` (someone clicked), that goal is used; otherwise one is picked that is
   well away from the old goal and off the route the swarm has learned, so the
   swarm has to search for it, and that makes a good session. */
function planMove(rand, world, values, agents, goal = null, tries = 6) {
  const fromStart = distances(world, world.start), fromGoal = distances(world, world.goal);
  const fromRoute = distancesFrom(world, believedRoute(world, values));
  const far = Math.max(world.W, world.H);
  const candidates = [];
  if (goal !== null) candidates.push(goal);
  else {
    for (const off of [3, 2, 0]) {
      for (let s = 0; s < fromStart.length; s++) {
        if (fromStart[s] >= 0.6 * fromStart[world.goal] && fromGoal[s] >= 0.3 * far && fromRoute[s] >= off) candidates.push(s);
      }
      if (candidates.length) break;
    }
  }
  let pick = null, cost = Infinity;
  for (let k = 0; k < tries && cost > 0 && candidates.length; k++) {
    const target = goal !== null ? goal : candidates[Math.floor(rand() * candidates.length)];
    const moved = withGoal(world, target);
    if (!moved) return null;
    const session = rehearse(moved, Math.floor(rand() * 2 ** 31), agents, { least: 2, values });
    const c = misfit(session, { firstLow: 0, firstHigh: 0.5, rounds: 4 });
    if (c < cost) { pick = session; cost = c; }
  }
  return pick;
}

/* The cells from which the swarm believes it knows a way: following the best
   neighbour from there keeps going downhill until a cell believed to be the
   goal. Returns the value of those cells, and -1 for every other cell. */
function knownRoutes(world, V) {
  const n = world.W * world.H, known = new Float32Array(n).fill(-1);
  const order = [];
  for (let s = 0; s < n; s++) if (!world.walls[s]) order.push(s);
  order.sort((a, b) => V[a] - V[b]);
  for (const s of order) {
    if (V[s] < 0.5) { known[s] = V[s]; continue; }
    let m = -1;
    for (const c of world.next[s]) if (m < 0 || V[c] < V[m]) m = c;
    if (m >= 0 && V[m] < V[s] && known[m] >= 0) known[s] = V[s];
  }
  return known;
}

if (typeof module !== "undefined") {
  module.exports = {
    SETTINGS, seeded, makeWorld, distances, shortestRoute, randomCourse, withGoal,
    Swarm, rehearse, misfit, planCourse, planMove, knownRoutes,
  };
}

/* ---------- View ---------- */

if (typeof document !== "undefined") {
  const boot = () => document.querySelectorAll("[data-band]").forEach(mountBand);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
}

function mountBand(section) {
  if (section.dataset.bandReady) return;
  section.dataset.bandReady = "swarm";
  section.classList.add("swarm-band");

  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const touch = window.matchMedia("(pointer: coarse)").matches;

  /* Pacing, in milliseconds. Cycles alternate: a new course learned from
     scratch (about 22 seconds), then the target moves and the swarm adapts
     (about 15). Round 1 is over within about 7 seconds of the band coming into view. */
  const CYCLE_MS = 22000;
  const GAP_MS = 220;     // between rounds: the stragglers fade, the table updates
  const MOVE_MS = 700;    // the target glides to its new place
  const COURSE_MS = 800;  // one course fades out and the next fades in
  const HOLD_MS = 500;    // a breath on a settled swarm
  const FADE_IN_MS = 260; // agents appearing at the start

  /* ---- DOM: a heading (unless the section brings its own), the stage, a status line ---- */
  const el = (tag, cls, html) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (html) node.innerHTML = html;
    return node;
  };
  // <section data-band data-band-plain> shows the maze alone: no heading, caption or controls.
  const plain = section.hasAttribute("data-band-plain");
  if (!plain && !section.children.length) {
    section.append(el("div", "sb-head",
      '<h2 class="sb-title">Learning by trial and error</h2>' +
      '<p class="sb-note"><span data-agents>Two hundred</span> agents share one table: how many steps they think each cell is from the target. ' +
      "After every round the table is updated from what happened, and the next round follows it.</p>"));
  }
  const stage = el("div", "sb-stage");
  const canvas = el("canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "A swarm of dots learning, round by round, to find its way through a maze to a target.");
  stage.append(canvas);
  const bars = el("span", "sb-bars");
  bars.setAttribute("aria-hidden", "true");
  const caption = el("p", "sb-caption");
  const status = el("div", "sb-status");
  status.append(bars, caption);
  const button = el("button", "sb-move", "Move the target");
  button.type = "button";
  const actions = el("p", "sb-act");
  actions.append(button, el("span", "sb-or", touch ? " or tap the maze" : " or click the maze"));
  const foot = el("div", "sb-foot");
  foot.append(status, actions);
  section.append(stage);
  if (!plain) section.append(foot);
  const agentCount = section.querySelector("[data-agents]");

  /* ---- Colours, from the site's tokens when present ---- */
  const css = getComputedStyle(section);
  const token = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
  const ACCENT = token("--accent", "#2c56a6");
  const AWARD = token("--award", "#9c6514");
  const INK = token("--ink", "#17191e");
  const WALL = "#dfe2e8";
  const DOT = "#d4d7de";

  /* ---- Layout ---- */
  const ctx = canvas.getContext("2d");
  const trace = document.createElement("canvas"); // where this round's agents have been, built up frame by frame
  const tctx = trace.getContext("2d");
  let cssW = 0, cssH = 0, dpr = 1, cols = 0, rows = 0, cell = 1, oy = 0;
  // The canvas reaches this far past the stage on every side, so round wall ends
  // and the target's halo can spill over the edge instead of being cut off.
  const BLEED = 18;

  /* Size the canvas to the stage and choose the grid: cells of up to 30-42 px that
     fill the width exactly. Returns true if the grid changed. A stage that
     grows a little keeps its grid, so the course on screen carries on. */
  function measure() {
    const r = stage.getBoundingClientRect();
    const stageW = Math.max(1, Math.round(r.width));
    const stageH = Math.max(1, Math.round(r.height));
    cssW = stageW + 2 * BLEED;
    cssH = stageH + 2 * BLEED;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    for (const c of [canvas, trace]) { c.width = Math.round(cssW * dpr); c.height = Math.round(cssH * dpr); }
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    // Cells shrink when the stage is short, so at least six rows always fit and nothing is cut off.
    const target = Math.min(stageW >= 900 ? 42 : stageW >= 600 ? 36 : 30, stageH / 6);
    const nextCols = Math.max(8, Math.ceil(stageW / target));
    cell = stageW / nextCols;
    let nextRows = Math.max(1, Math.floor(stageH / cell));
    if (nextCols === cols && nextRows > rows && nextRows - rows <= 2) nextRows = rows;
    oy = (stageH - nextRows * cell) / 2;
    const changed = nextCols !== cols || nextRows !== rows;
    cols = nextCols;
    rows = nextRows;
    return changed;
  }
  const px = (g) => BLEED + g * cell;     // grid units (cell centres at i + 0.5) to CSS pixels
  const py = (g) => BLEED + oy + g * cell;
  const cx = (s) => (s % cols) + 0.5;
  const cy = (s) => Math.floor(s / cols) + 0.5;

  /* ---- The plan: a queue of acts (rounds, a moving target, a new course, pauses) ---- */
  const rand = Math.random;
  let acts = [], current = null, elapsed = 0;
  let world = null;          // the walls on screen
  let goal = { x: 0, y: 0 }; // where the target is drawn, in grid units
  let knowledge = null;      // the swarm's values at this point of the replay
  let known = null;          // ... and the cells from which it believes it knows the way
  let fade = 1;              // scene opacity, while changing courses
  let best = 0, stepMs = 90, lastU = 0;
  let roundNumber = 0, nextKind = "course";
  const offsets = Array.from({ length: 220 }, () => [(rand() - 0.5) * 0.5, (rand() - 0.5) * 0.5]); // each agent's place within a cell

  const agentsFor = () => Math.max(70, Math.min(220, Math.round(cols * rows * 0.3)));

  function believe(values) {
    knowledge = values;
    known = knownRoutes(world, values);
  }

  /* Turn a rehearsed round into what the view needs: each agent's path in grid
     units (ending exactly on the target if it got there) and when it arrived. */
  function roundAct(session, round, number) {
    const w = session.world, agents = [];
    round.walks.forEach((walk, a) => {
      const [dx, dy] = offsets[a];
      const n = walk.length, xs = new Float32Array(n), ys = new Float32Array(n);
      for (let i = 0; i < n; i++) { xs[i] = cx(walk[i]) + dx; ys[i] = cy(walk[i]) + dy; }
      const arrived = walk[n - 1] === w.goal;
      if (arrived) { xs[n - 1] = cx(w.goal); ys[n - 1] = cy(w.goal); }
      agents.push({ xs, ys, end: n - 1, arrived });
    });
    const arrivals = agents.filter((g) => g.arrived).map((g) => g.end).sort((a, b) => a - b);
    return {
      kind: "round", number, session, round, agents, arrivals,
      before: knownRoutes(w, round.before), after: knownRoutes(w, round.after),
      duration: () => round.steps * stepMs + GAP_MS,
    };
  }

  /* The rounds of a session, then a pause. */
  const sessionActs = (session, first) => [
    ...session.rounds.map((round, k) => roundAct(session, round, first + k)),
    { kind: "hold", duration: () => HOLD_MS },
  ];

  /* Pace a new course to last about CYCLE_MS: what is left after the fixed
     pauses is shared out over all the steps, within calm limits. */
  function pace(list) {
    let fixed = 0, steps = 0;
    for (const a of list) {
      if (a.kind === "round") { steps += a.round.steps; fixed += GAP_MS; }
      else fixed += a.kind === "course" ? COURSE_MS : HOLD_MS;
    }
    stepMs = Math.max(70, Math.min(100, (CYCLE_MS - fixed) / Math.max(1, steps)));
  }

  /* The next cycle: either a new course learned from scratch, or the same
     course with the target moved, learned on top of what the swarm knows. */
  function planCycle(first) {
    let list = [];
    if (nextKind === "move" && world) {
      const moved = planMove(rand, world, knowledge, agentsFor());
      if (moved) list = [{ kind: "move", session: moved, duration: () => MOVE_MS }, ...sessionActs(moved, roundNumber + 1)];
    }
    if (!list.length) {
      const course = planCourse(rand, cols, rows, agentsFor());
      list = [{ kind: "course", session: course, duration: () => COURSE_MS }, ...sessionActs(course, 1)];
      pace(list);
      if (first) { list.shift(); enterCourse(course); }
      nextKind = "move";
    } else nextKind = "course";
    acts = list;
  }

  function enterCourse(session) {
    world = session.world;
    tctx.clearRect(0, 0, trace.width, trace.height);
    goal = { x: cx(world.goal), y: cy(world.goal) };
    believe(session.rounds[0].before);
    bars.textContent = "";
    if (agentCount) agentCount.textContent = String(session.agents);
  }

  function startAct(next) {
    current = next;
    elapsed = 0;
    if (current.kind === "round") {
      if (current.number === 1) enterCourse(current.session);
      roundNumber = current.number;
      world = current.session.world;
      best = current.session.best;
      knowledge = current.round.before;
      known = current.before;
      lastU = 0;
      tctx.clearRect(0, 0, trace.width, trace.height);
      const previous = bars.lastElementChild;
      if (previous && previous.dataset.goal !== String(world.goal)) bars.append(el("b")); // marks where the target moved
      current.bar = el("i");
      current.bar.dataset.goal = String(world.goal);
      bars.append(current.bar);
      describeRound();
    } else if (current.kind === "move") {
      current.from = { ...goal };
      current.to = { x: cx(current.session.world.goal), y: cy(current.session.world.goal) };
      // Does the swarm already know a way to the old spot? Then it will look there first.
      const believes = known && known[world.start] >= 0;
      caption.textContent = (current.byVisitor ? "You moved the target." : "The target moved.") +
        (believes ? " The swarm still believes it is where it was." : "");
    } else if (current.kind === "course") {
      caption.textContent = "A new course. The table starts empty.";
    }
  }

  /* ---- Advancing time ---- */
  function advance(dt) {
    elapsed += dt;
    if (current.kind === "move") {
      const t = Math.min(1, elapsed / MOVE_MS), e = t * t * (3 - 2 * t);
      goal = { x: current.from.x + (current.to.x - current.from.x) * e, y: current.from.y + (current.to.y - current.from.y) * e };
    } else if (current.kind === "course") {
      const half = COURSE_MS / 2;
      fade = elapsed < half ? 1 - elapsed / half : Math.min(1, (elapsed - half) / half);
      if (elapsed >= half && world !== current.session.world) enterCourse(current.session);
    } else if (current.kind === "round") {
      if (elapsed >= current.round.steps * stepMs && knowledge !== current.round.after) {
        knowledge = current.round.after; // the round is over: the table takes in what happened
        known = current.after;
      }
      describeRound();
    }
    if (elapsed >= current.duration()) {
      fade = 1;
      if (!acts.length) planCycle(false);
      startAct(acts.shift());
    }
  }

  /* Caption and learning-curve bar for the round on screen. */
  function describeRound() {
    const a = current, u = elapsed / stepMs, total = a.agents.length;
    let k = 0;
    while (k < a.arrivals.length && a.arrivals[k] <= u) k++;
    a.bar.style.height = Math.max(1, Math.round((k / total) * 100)) + "%";
    let text = `Round ${a.number} · ${k} of ${total} reached the target`;
    if (u >= a.round.steps && k) text += `, in ${Math.round(a.round.meanSteps)} steps on average (shortest ${best})`;
    if (caption.textContent !== text) caption.textContent = text;
  }

  /* Someone asked for the target somewhere else (`target` is a cell, or null
     for anywhere suitable): carry on from what the swarm knows now. */
  function moveTargetTo(target) {
    if (paused) setPaused(false); // moving the target means the visitor wants to watch
    if (!world) return;
    const moved = planMove(rand, world, knowledge, agentsFor(), target, target === null ? 6 : 2);
    if (!moved) { nudge(); return; }
    // A round cut short is dropped; the next one takes its number.
    let number = roundNumber + 1;
    if (current.kind === "round" && elapsed < current.round.steps * stepMs) {
      number = current.number;
      current.bar.remove();
    }
    acts = [{ kind: "move", session: moved, byVisitor: true, duration: () => MOVE_MS }, ...sessionActs(moved, number)];
    acts[acts.length - 1].duration = () => HOLD_MS * 2.5; // linger on what the visitor did
    nextKind = "course";
    fade = 1;
    if (still) { showStill(); return; }
    startAct(acts.shift());
    wake();
  }

  /* ---- Drawing ---- */

  /* Position along a path at time u (in steps), on a smooth curve through the cells. */
  function along(v, u, end) {
    if (u >= end) return v[end];
    if (u <= 0) return v[0];
    const i = Math.floor(u), f = u - i;
    const p0 = v[Math.max(0, i - 1)], p1 = v[i], p2 = v[i + 1], p3 = v[Math.min(end, i + 2)];
    return 0.5 * (2 * p1 + (p2 - p0) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (3 * p1 - p0 - 3 * p2 + p3) * f * f * f);
  }

  /* Walls are thick rounded strokes joining neighbouring wall cells; walls on
     the edge of the field reach out to the edge. */
  function drawWalls() {
    const { W, H, walls } = world;
    const line = (x1, y1, x2, y2) => { ctx.moveTo(px(x1), py(y1)); ctx.lineTo(px(x2), py(y2)); };
    ctx.strokeStyle = WALL;
    ctx.lineWidth = cell * 0.34;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let s = 0; s < W * H; s++) {
      if (!walls[s]) continue;
      const x = s % W, y = (s - x) / W, gx = x + 0.5, gy = y + 0.5;
      line(gx, gy, gx, gy);
      if (x + 1 < W && walls[s + 1]) line(gx, gy, gx + 1, gy);
      if (y + 1 < H && walls[s + W]) line(gx, gy, gx, gy + 1);
      if (y === 0) line(gx, gy, gx, 0);
      if (y === H - 1) line(gx, gy, gx, H);
      if (x === 0) line(gx, gy, 0, gy);
      if (x === W - 1) line(gx, gy, W, gy);
    }
    ctx.stroke();
  }

  /* A dot in every open cell: grey where the swarm knows nothing yet, blue
     where it believes it knows the way, deeper the closer to the target. */
  function drawBeliefs(from, to, mix) {
    const { W, H, walls } = world;
    let far = 1;
    for (let s = 0; s < W * H; s++) far = Math.max(far, from[s], to[s]);
    const r0 = Math.max(1, cell * 0.03);
    ctx.fillStyle = DOT;
    ctx.beginPath();
    for (let s = 0; s < W * H; s++) {
      if (walls[s]) continue;
      ctx.moveTo(px(cx(s)) + r0, py(cy(s)));
      ctx.arc(px(cx(s)), py(cy(s)), r0, 0, Math.PI * 2);
    }
    ctx.fill();
    ctx.fillStyle = ACCENT;
    for (let s = 0; s < W * H; s++) {
      if (walls[s]) continue;
      const a = from[s] >= 0 ? 1 - from[s] / (far + 4) : 0;
      const b = to[s] >= 0 ? 1 - to[s] / (far + 4) : 0;
      const near = a + (b - a) * mix;
      if (near <= 0.01) continue;
      ctx.globalAlpha = fade * (0.15 + 0.35 * near) * Math.min(1, near * 6);
      ctx.beginPath();
      ctx.arc(px(cx(s)), py(cy(s)), r0 + cell * 0.05 * near, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = fade;
  }

  function drawStartAndGoal(pulse) {
    const s = world.start;
    ctx.strokeStyle = INK;
    ctx.globalAlpha = fade * 0.28;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.arc(px(cx(s)), py(cy(s)), cell * 1.25, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    const gx = px(goal.x), gy = py(goal.y);
    ctx.fillStyle = AWARD;
    ctx.globalAlpha = fade * (0.1 + 0.12 * pulse);
    ctx.beginPath();
    ctx.arc(gx, gy, cell * (0.62 + 0.18 * pulse), 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = fade;
    ctx.beginPath();
    ctx.arc(gx, gy, cell * 0.27, 0, Math.PI * 2);
    ctx.fill();
  }

  /* Faint lines where the agents went: the trace of the round so far. */
  function extendTrace(a, from, to) {
    tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    tctx.strokeStyle = ACCENT;
    tctx.globalAlpha = 0.035;
    tctx.lineWidth = 1.2;
    tctx.beginPath();
    for (const g of a.agents) {
      if (from >= g.end) continue;
      tctx.moveTo(px(along(g.xs, from, g.end)), py(along(g.ys, from, g.end)));
      for (let v = Math.min(to, from + 0.25); ; v = Math.min(to, v + 0.25)) {
        tctx.lineTo(px(along(g.xs, v, g.end)), py(along(g.ys, v, g.end)));
        if (v >= to) break;
      }
    }
    tctx.stroke();
  }

  /* The agents at time u (in steps): a dot with a short tail each. Arrivals
     sink into the target; agents that ran out of steps fade where they stopped. */
  const TAIL = [0, 0.35, 0.7, 1.05, 1.4, 1.75];
  function drawAgents(a, u, gapT) {
    const r = Math.max(1.6, cell * 0.075);
    const appear = Math.min(1, (u * stepMs) / FADE_IN_MS);
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = r * 1.1;
    ctx.lineCap = "round";
    for (let k = 1; k < TAIL.length; k++) {
      ctx.globalAlpha = fade * appear * 0.32 * (1 - k / TAIL.length);
      ctx.beginPath();
      for (const g of a.agents) {
        if (u - TAIL[k - 1] >= g.end || u - TAIL[k] < 0) continue;
        ctx.moveTo(px(along(g.xs, u - TAIL[k - 1], g.end)), py(along(g.ys, u - TAIL[k - 1], g.end)));
        ctx.lineTo(px(along(g.xs, u - TAIL[k], g.end)), py(along(g.ys, u - TAIL[k], g.end)));
      }
      ctx.stroke();
    }
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    const fading = [];
    for (const g of a.agents) {
      let alpha = appear;
      if (u >= g.end) alpha = g.arrived ? 1 - (u - g.end) / 3 : 0.55 * (1 - gapT);
      if (alpha <= 0) continue;
      const x = px(along(g.xs, u, g.end)), y = py(along(g.ys, u, g.end));
      if (alpha !== appear) { fading.push([x, y, alpha]); continue; }
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, Math.PI * 2);
    }
    ctx.globalAlpha = fade * appear;
    ctx.fill();
    for (const [x, y, alpha] of fading) {
      ctx.globalAlpha = fade * alpha;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = fade;
  }

  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.globalAlpha = fade;
    if (current.kind !== "round") {
      drawBeliefs(known, known, 0);
      drawWalls();
      drawStartAndGoal(0);
      // The last round's trail stays through a pause, and fades while the target moves.
      ctx.globalAlpha = fade * (current.kind === "move" ? Math.max(0, 1 - elapsed / MOVE_MS) : 1);
      ctx.drawImage(trace, 0, 0, cssW, cssH);
      return;
    }
    const a = current, moving = a.round.steps * stepMs;
    const u = Math.min(elapsed, moving) / stepMs;
    const gapT = Math.min(1, Math.max(0, (elapsed - moving) / GAP_MS));
    drawBeliefs(a.before, a.after, Math.min(1, gapT * 1.5));
    drawWalls();
    let pulse = 0; // recent arrivals make the target swell a little
    for (let k = a.arrivals.length - 1; k >= 0 && a.arrivals[k] > u - 6; k--) if (a.arrivals[k] <= u) pulse += 0.12;
    drawStartAndGoal(Math.min(1, pulse));
    if (u > lastU) { extendTrace(a, lastU, u); lastU = u; }
    const pauseNext = acts.length && acts[0].kind === "hold"; // keep the trail for the pause after a settled round
    ctx.globalAlpha = fade * (pauseNext ? 1 : 1 - gapT);
    ctx.drawImage(trace, 0, 0, cssW, cssH);
    drawAgents(a, u, gapT);
  }

  /* ---- Reduced motion: one still frame of the swarm once it has settled ---- */
  function showStill() {
    const last = [current, ...acts].filter((a) => a && a.kind === "round").pop();
    current = last;
    acts = [];
    world = last.session.world;
    best = last.session.best;
    goal = { x: cx(world.goal), y: cy(world.goal) };
    knowledge = last.round.after;
    known = last.after;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.globalAlpha = 1;
    drawBeliefs(known, known, 0);
    drawWalls();
    drawStartAndGoal(0);
    // Every route walked in that round, faintly, and the agents spread out along them.
    ctx.strokeStyle = ACCENT;
    ctx.globalAlpha = 0.06;
    ctx.lineWidth = 1.2;
    for (const g of last.agents) {
      ctx.beginPath();
      ctx.moveTo(px(g.xs[0]), py(g.ys[0]));
      for (let v = 0.25; v <= g.end; v += 0.25) ctx.lineTo(px(along(g.xs, v, g.end)), py(along(g.ys, v, g.end)));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const r = Math.max(1.6, cell * 0.075);
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    last.agents.forEach((g, i) => {
      const u = ((i * 0.618) % 1) * g.end;
      const x = px(along(g.xs, u, g.end)), y = py(along(g.ys, u, g.end));
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, Math.PI * 2);
    });
    ctx.fill();
    bars.textContent = "";
    for (const round of last.session.rounds) {
      const bar = el("i");
      bar.style.height = Math.max(1, Math.round(round.share * 100)) + "%";
      bars.append(bar);
    }
    caption.textContent = `Round ${last.number} · ${last.round.reached} of ${last.agents.length} reached the target, ` +
      `in ${Math.round(last.round.meanSteps)} steps on average (shortest ${best})`;
  }

  /* ---- Running, pausing, resizing ---- */
  let visible = false, raf = 0, previous = 0, paused = false;

  // The whole animation runs at this fraction of real time: calmer than the pacing
  // constants above on their own, chosen by eye.
  const SPEED = 0.65;
  function frame(now) {
    raf = 0;
    if (!visible || document.hidden || paused) { previous = 0; return; }
    const dt = previous ? Math.min(now - previous, 50) : 16;
    previous = now;
    advance(dt * SPEED);
    draw();
    raf = requestAnimationFrame(frame);
  }
  function wake() { if (!raf && !still && !paused && current && visible && !document.hidden) raf = requestAnimationFrame(frame); }

  /* ---- Pause and play. The page may supply its own button ([data-band-pause] anywhere);
     otherwise a small one sits in the maze's corner. ---- */
  const PAUSE_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="4" y="3" width="2.6" height="10" rx="1"/><rect x="9.4" y="3" width="2.6" height="10" rx="1"/></svg>';
  const PLAY_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3.2v9.6a.8.8 0 0 0 1.2.7l7.6-4.8a.8.8 0 0 0 0-1.4L6.2 2.5A.8.8 0 0 0 5 3.2Z"/></svg>';
  const supplied = document.querySelector("[data-band-pause]");
  const toggle = supplied || el("button", "sb-pause");
  toggle.type = "button";
  function setPaused(value) {
    paused = value;
    toggle.innerHTML = paused ? PLAY_ICON : PAUSE_ICON;
    toggle.setAttribute("aria-label", paused ? "Play the animation" : "Pause the animation");
    toggle.title = paused ? "Play" : "Pause";
    if (!paused) wake();
  }
  toggle.addEventListener("click", () => setPaused(!paused));
  if (still) toggle.hidden = true; // nothing moves, so there is nothing to pause
  else {
    if (!supplied) stage.append(toggle);
    setPaused(false);
  }

  function restart() {
    fade = 1;
    nextKind = "course";
    planCycle(true);
    if (still) { showStill(); return; }
    startAct(acts.shift());
    draw();
    wake();
  }

  let resizeTimer = 0;
  new ResizeObserver(() => {
    const changed = measure();
    if (changed || !world) {
      // A new grid size needs a new course; wait until the resizing settles.
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(restart, world ? 200 : 0);
    } else if (still) showStill();
    else { tctx.clearRect(0, 0, trace.width, trace.height); lastU = 0; draw(); }
  }).observe(stage);

  // Play only while a good part of the maze is in view, so the first round is not spent off-screen.
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting && entry.intersectionRatio >= 0.3;
    wake();
  }, { threshold: [0, 0.3] }).observe(stage);
  document.addEventListener("visibilitychange", wake);

  /* ---- Interaction: click or tap a cell to move the target there ---- */
  function nudge() {
    stage.classList.remove("sb-nope");
    void stage.offsetWidth; // restart the little shake
    stage.classList.add("sb-nope");
  }
  canvas.addEventListener("click", (event) => {
    if (!world) return;
    const r = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - r.left - BLEED) / cell), y = Math.floor((event.clientY - r.top - BLEED - oy) / cell);
    if (x < 0 || y < 0 || x >= cols || y >= rows) return;
    const s = y * cols + x;
    if (s === world.goal || world.walls[s]) { nudge(); return; }
    moveTargetTo(s);
  });
  button.addEventListener("click", () => moveTargetTo(null));
}
