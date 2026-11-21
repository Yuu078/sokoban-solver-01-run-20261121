// Sokoban solver for make.fun's "Build the best Sokoban solver, 21 November 2026".
//
// Approach: the generator lays one hallway across the board with one to four
// rooms off it, each holding 3-4 boxes and as many goals, and a box pushed out
// of its room can never come back - so every room is solved on its own, in
// push-optimal order, and the room optima add up to the level's. Each room is
// searched with A* over pushes: a state is the boxes' squares plus the
// player's reachable region, the cost is pushes made, and the estimate is the
// cheapest assignment of boxes to goals by floor distance (admissible: one
// push moves one box one square, so it cuts the estimate by at most one).
// Two deadlocks are pruned, both provably safe: a box on a square from which
// no sequence of pushes (ignoring the other boxes) reaches any goal, and a
// box frozen inside a 2x2 block of walls and boxes with a box off its goal.
// A plain push-BFS with the same pruning is the fallback when A* exhausts
// its state budget. Every budget is counted in states, never the clock, so
// two runs answer byte for byte alike.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Work caps: counted in states, never the clock, so runs are deterministic. */
const ASTAR_STATES = 300_000;
const BFS_STATES = 300_000;
const INSTANCE_STATES = 2_000_000;

const DIRS: ReadonlyArray<readonly [number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Budget {
  used: number;
  limit: number;
}

class BudgetOut extends Error {}

/** A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player, + player on a goal). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * The squares from which a box alone can be pushed onto some goal: pull back
 * from the goals over open floor (a pull needs the player to stand beyond the
 * box, so both the square and the one past it must be floor).
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (wall[y] === 0 && wall[y - d] === 0 && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Floor distances from every goal: dist[g][cell], Infinity where unreachable. */
function goalDistances(level: Level): number[][] {
  const { w, wall, goals } = level;
  const size = wall.length;
  const all: number[][] = [];
  for (const g of goals) {
    const dist = new Array<number>(size).fill(Infinity);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (wall[q] === 0 && dist[q] === Infinity) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
    all.push(dist);
  }
  return all;
}

/**
 * Cheapest assignment of boxes to goals by floor distance. Exact by
 * permutation for small rooms; for larger sets the sum of each box's nearest
 * goal, still admissible (one push cuts either sum by at most one).
 */
function assignmentEstimate(boxes: number[], dist: number[][], goals: number[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 6) {
    let s = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (let g = 0; g < goals.length; g++) m = Math.min(m, dist[g]![b]!);
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  }
  // Exact minimum over the matchings, by walking the permutations.
  const used = new Array<boolean>(goals.length).fill(false);
  let best = Infinity;
  const walk = (bi: number, acc: number): void => {
    if (acc >= best) return;
    if (bi === n) {
      best = acc;
      return;
    }
    for (let g = 0; g < goals.length; g++) {
      if (used[g]) continue;
      const d = dist[g]![boxes[bi]!]!;
      if (d === Infinity) continue;
      used[g] = true;
      walk(bi + 1, acc + d);
      used[g] = false;
    }
  };
  walk(0, 0);
  return best;
}

/** The squares the player reaches from `from` with `occupied` in the way; returns the lowest square reached. */
function reach(level: Level, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (first direction first on ties). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

interface Solved {
  pushes: number;
  moves: string;
}

/**
 * Push-optimal search over one board: A* when `astar` (heuristic on), plain
 * push-BFS when off (f = pushes, heap order by insertion). A state is the
 * boxes' squares plus the player's reachable region, named by its lowest
 * square. Returns null when the state cap runs out or the board is unsolved.
 */
function solvePush(level: Level, maxStates: number, astar: boolean, budget: Budget): Solved | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const dist = goalDistances(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);

  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // The states in order found: boxes, player square, parent, push from/to, cost, estimate.
  const sBoxes: number[][] = [level.boxes];
  const sPlayer: number[] = [level.player];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sG: number[] = [0];
  const sH: number[] = [astar ? assignmentEstimate(level.boxes, dist, goals) : 0];
  const sOnGoal: number[] = [level.boxes.filter((b) => goal[b]).length];
  // best maps a state's key to the cheapest pushes known to reach it.
  const best = new Map<string, number>();
  const occupied = new Uint8Array(size);
  for (const b of level.boxes) occupied[b] = 1;
  const startLow = reach(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  const keyOf = (boxes: number[], low: number): string => `${low}|${boxes.join(',')}`;
  best.set(keyOf(level.boxes, startLow), 0);

  // Binary heap on (f, -onGoal, sequence): f first, then more boxes home, then oldest.
  const heap: number[] = [];
  let seq = 0;
  const seqOf = new Map<number, number>();
  const less = (a: number, b: number): boolean => {
    const fa = sG[a]! + sH[a]!;
    const fb = sG[b]! + sH[b]!;
    if (fa !== fb) return fa < fb;
    if (sOnGoal[a] !== sOnGoal[b]) return sOnGoal[a]! > sOnGoal[b]!;
    return seqOf.get(a)! < seqOf.get(b)!;
  };
  const heapPush = (s: number): void => {
    seqOf.set(s, seq++);
    heap.push(s);
    for (let i = heap.length - 1; i > 0; ) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  heapPush(0);

  let found = solved(level.boxes) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    const s = heapPop();
    const boxes = sBoxes[s]!;
    // The key needs the region; recompute it from the state's player square.
    // Skip stale heap entries: a cheaper path to this state was recorded.
    for (const b of boxes) occupied[b] = 1;
    const low = reach(level, occupied, sPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    if (best.get(keyOf(boxes, low)) !== sG[s]) {
      for (const b of boxes) occupied[b] = 0;
      continue;
    }
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let next: number[] | null = null;
        let key = '';
        let h = 0;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          const nLow = reach(level, occupied, b, mark, ++stamp, stack);
          key = keyOf(next, nLow);
          h = astar ? assignmentEstimate(next, dist, goals) : 0;
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        if (h === Infinity) continue;
        const prevG = best.get(key);
        if (prevG !== undefined && prevG <= sG[s]! + 1) continue;
        if (budget.used >= budget.limit || sBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          throw new BudgetOut();
        }
        budget.used++;
        const ns = sBoxes.length;
        best.set(key, sG[s]! + 1);
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        sG.push(sG[s]! + 1);
        sH.push(h);
        sOnGoal.push(next.filter((x) => goal[x]).length);
        heapPush(ns);
        if (solved(next)) {
          found = ns;
          break;
        }
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  // Rebuild the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = sBoxes[s]!;
    const from = sFrom[s]!;
    const to = sTo[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds a box is a room. Null when some part holds more or fewer boxes than
 * goals, or a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The level solved room by room, the other rooms' boxes standing as walls; the whole board when it is no set of rooms. */
function solveLevel(level: Level, budget: Budget): string {
  const rooms = roomsOf(level);
  if (rooms === null) {
    try {
      const whole = solvePush(level, ASTAR_STATES, true, budget) ?? solvePush(level, BFS_STATES, false, budget);
      return whole === null ? '' : whole.moves;
    } catch (e) {
      if (e instanceof BudgetOut) return '';
      throw e;
    }
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    let solved: Solved | null = null;
    try {
      solved = solvePush(sub, ASTAR_STATES, true, budget) ?? solvePush(sub, BFS_STATES, false, budget);
    } catch (e) {
      if (e instanceof BudgetOut) return '';
      throw e;
    }
    if (solved === null) return '';
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    const budget: Budget = { used: 0, limit: INSTANCE_STATES };
    moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')), budget);
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
