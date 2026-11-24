import {
  COLS,
  applyMove,
  encodeObservation,
  getLegalCols,
  type GameState,
} from './c4';

export interface PolicyValueEvaluator {
  evaluate(state: GameState): Promise<{ logits: Float32Array; value: number }>;
}

export interface SearchAnalysis {
  bestCol: number;
  /** Column probability distribution (7 values summing to 1 over legal columns) */
  policyProbs: number[];
  /** Raw network prior probabilities before tree search */
  rawPriors: number[];
  /** Win probability in [0, 1] from Player 1 (Red) perspective */
  p1WinProb: number;
  /** Value in [-1, 1] from the current player-to-move's perspective */
  currentPlayerValue: number;
  /** Visit counts per column from MCTS */
  visits: number[];
  /** Total simulations executed */
  simulations: number;
  /** Elapsed search time in milliseconds */
  elapsedMs: number;
}

export function softmaxMasked(logits: Float32Array, legalCols: number[]): number[] {
  const probs = new Array<number>(COLS).fill(0);
  if (legalCols.length === 0) return probs;

  let maxL = -Infinity;
  for (const c of legalCols) {
    if (logits[c] > maxL) maxL = logits[c];
  }
  let sum = 0;
  for (const c of legalCols) {
    const e = Math.exp(logits[c] - maxL);
    probs[c] = e;
    sum += e;
  }
  if (sum > 0) {
    for (const c of legalCols) probs[c] /= sum;
  } else {
    const uniform = 1 / legalCols.length;
    for (const c of legalCols) probs[c] = uniform;
  }
  return probs;
}

class TreeNode {
  state: GameState;
  legalCols: number[];
  isTerminal: boolean;
  terminalValue: number; // from perspective of player TO MOVE at this node
  expanded = false;
  priors: number[] = new Array(COLS).fill(0);
  visits: number[] = new Array(COLS).fill(0);
  valueSum: number[] = new Array(COLS).fill(0);
  children: Map<number, TreeNode> = new Map();

  constructor(state: GameState) {
    this.state = state;
    this.isTerminal = state.winner !== 0 || state.isDraw;
    if (state.winner !== 0) {
      // Previous mover won, so current player to move lost
      this.terminalValue = -1.0;
    } else {
      this.terminalValue = 0.0;
    }
    this.legalCols = this.isTerminal ? [] : getLegalCols(state.board);
  }
}

function selectPuct(node: TreeNode, cPuct: number): number {
  let totalVisits = 0;
  for (const c of node.legalCols) totalVisits += node.visits[c];
  const sqrtTotal = Math.sqrt(Math.max(1, totalVisits));

  let bestCol = node.legalCols[0];
  let bestScore = -Infinity;
  for (const c of node.legalCols) {
    const n = node.visits[c];
    const q = n > 0 ? node.valueSum[c] / n : 0.0;
    const u = (cPuct * node.priors[c] * sqrtTotal) / (1 + n);
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      bestCol = c;
    }
  }
  return bestCol;
}

function backup(path: Array<{ node: TreeNode; action: number }>, leafValue: number): void {
  let v = -leafValue;
  for (let i = path.length - 1; i >= 0; i--) {
    const { node, action } = path[i];
    node.visits[action] += 1;
    node.valueSum[action] += v;
    v = -v;
  }
}

export function valueToP1WinProb(valueFromCurrentPlayer: number, state: GameState): number {
  // Value is in [-1, 1] from currentPlayer's perspective. Convert to P1 perspective, then map [-1,1] -> [0,1].
  const p1Val = state.currentPlayer === 1 ? valueFromCurrentPlayer : -valueFromCurrentPlayer;
  return Math.max(0, Math.min(1, 0.5 * (p1Val + 1)));
}

export async function runMctsSearch(
  state: GameState,
  evaluator: PolicyValueEvaluator,
  simulations: number,
  cPuct = 1.5,
  shouldAbort?: () => boolean,
): Promise<SearchAnalysis> {
  const t0 = performance.now();

  if (state.winner !== 0 || state.isDraw) {
    const curWin = state.winner === 0 ? 0 : state.winner === state.currentPlayer ? 1 : -1;
    return {
      bestCol: -1,
      policyProbs: new Array(COLS).fill(0),
      rawPriors: new Array(COLS).fill(0),
      p1WinProb: state.winner === 1 ? 1 : state.winner === -1 ? 0 : 0.5,
      currentPlayerValue: curWin,
      visits: new Array(COLS).fill(0),
      simulations: 0,
      elapsedMs: 0,
    };
  }

  const root = new TreeNode(state);
  const { logits, value: rootNetVal } = await evaluator.evaluate(state);
  root.priors = softmaxMasked(logits, root.legalCols);
  root.expanded = true;

  if (simulations <= 0 || root.legalCols.length === 1) {
    let bestCol = root.legalCols[0];
    let bestP = -1;
    for (const c of root.legalCols) {
      if (root.priors[c] > bestP) {
        bestP = root.priors[c];
        bestCol = c;
      }
    }
    const probs =
      root.legalCols.length === 1
        ? root.legalCols.reduce((arr, c) => {
            arr[c] = 1;
            return arr;
          }, new Array<number>(COLS).fill(0))
        : root.priors;
    return {
      bestCol,
      policyProbs: probs,
      rawPriors: root.priors.slice(),
      p1WinProb: valueToP1WinProb(rootNetVal, state),
      currentPlayerValue: rootNetVal,
      visits: new Array(COLS).fill(0),
      simulations: 0,
      elapsedMs: performance.now() - t0,
    };
  }

  let completedSims = 0;
  for (let s = 0; s < simulations; s++) {
    if (shouldAbort?.()) break;
    let cur = root;
    const path: Array<{ node: TreeNode; action: number }> = [];

    while (cur.expanded && !cur.isTerminal) {
      const action = selectPuct(cur, cPuct);
      path.push({ node: cur, action });
      let child = cur.children.get(action);
      if (!child) {
        child = new TreeNode(applyMove(cur.state, action));
        cur.children.set(action, child);
      }
      cur = child;
    }

    if (cur.isTerminal) {
      backup(path, cur.terminalValue);
    } else {
      const res = await evaluator.evaluate(cur.state);
      cur.priors = softmaxMasked(res.logits, cur.legalCols);
      cur.expanded = true;
      backup(path, res.value);
    }
    completedSims++;
  }

  const totalVisits = root.visits.reduce((a, b) => a + b, 0);
  const policyProbs = new Array<number>(COLS).fill(0);
  let bestCol = root.legalCols[0];
  let bestVisits = -1;
  let rootQWeighted = 0;

  if (totalVisits > 0) {
    for (const c of root.legalCols) {
      const v = root.visits[c];
      policyProbs[c] = v / totalVisits;
      rootQWeighted += (v / totalVisits) * (v > 0 ? root.valueSum[c] / v : 0);
      if (v > bestVisits) {
        bestVisits = v;
        bestCol = c;
      }
    }
  } else {
    for (const c of root.legalCols) {
      policyProbs[c] = root.priors[c];
    }
    rootQWeighted = rootNetVal;
  }

  const blendedVal = totalVisits > 0 ? rootQWeighted : rootNetVal;
  return {
    bestCol,
    policyProbs,
    rawPriors: root.priors.slice(),
    p1WinProb: valueToP1WinProb(blendedVal, state),
    currentPlayerValue: blendedVal,
    visits: root.visits.slice(),
    simulations: completedSims,
    elapsedMs: performance.now() - t0,
  };
}

export { encodeObservation };
