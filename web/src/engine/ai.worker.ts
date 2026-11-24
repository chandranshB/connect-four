import * as ort from 'onnxruntime-web/wasm';
import {
  COLS,
  ROWS,
  encodeObservation,
  type GameState,
} from './c4';
import { runMctsSearch, type PolicyValueEvaluator, type SearchAnalysis } from './mcts';

export type ModelTier = 'early' | 'mid' | 'pro';

interface WorkerAnalyzeRequest {
  type: 'analyze';
  requestId: number;
  tier: ModelTier;
  simulations: number;
  board: number[];
  ply: number;
  currentPlayer: 1 | -1;
  winner: 1 | -1 | 0;
  isDraw: boolean;
}

interface WorkerPreloadRequest {
  type: 'preload';
  tier: ModelTier;
}

type WorkerRequest = WorkerAnalyzeRequest | WorkerPreloadRequest;

const baseUrl = (import.meta.env.BASE_URL || '/').replace(/\/$/, '') + '/';

ort.env.wasm.wasmPaths = `${baseUrl}ort/`;
ort.env.wasm.numThreads = 1;

const sessionCache = new Map<ModelTier, Promise<ort.InferenceSession>>();
let latestRequestId = 0;

function loadSession(tier: ModelTier): Promise<ort.InferenceSession> {
  let cached = sessionCache.get(tier);
  if (!cached) {
    cached = ort.InferenceSession.create(`${baseUrl}models/${tier}.onnx`, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }).catch((err) => {
      sessionCache.delete(tier);
      throw err;
    });
    sessionCache.set(tier, cached);
  }
  return cached;
}

function reconstructState(req: WorkerAnalyzeRequest): GameState {
  return {
    board: Int8Array.from(req.board),
    ply: req.ply,
    currentPlayer: req.currentPlayer,
    winner: req.winner,
    isDraw: req.isDraw,
    winningLine: [],
    lastMove: null,
  };
}

function makeEvaluator(session: ort.InferenceSession): PolicyValueEvaluator {
  return {
    async evaluate(state: GameState) {
      const obs = encodeObservation(state);
      const tensor = new ort.Tensor('float32', obs, [1, 3, ROWS, COLS]);
      const outputs = await session.run({ board: tensor });
      const logitsData = outputs.policy_logits.data as Float32Array;
      const valueData = outputs.value.data as Float32Array;
      return {
        logits: new Float32Array(logitsData),
        value: Number(valueData[0]),
      };
    },
  };
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  if (msg.type === 'preload') {
    try {
      await loadSession(msg.tier);
      self.postMessage({ type: 'ready', tier: msg.tier });
    } catch (err) {
      self.postMessage({
        type: 'error',
        requestId: -1,
        message: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (msg.type === 'analyze') {
    latestRequestId = Math.max(latestRequestId, msg.requestId);
    try {
      const session = await loadSession(msg.tier);
      if (msg.requestId !== latestRequestId) return;

      const state = reconstructState(msg);
      const evaluator = makeEvaluator(session);
      const analysis: SearchAnalysis = await runMctsSearch(
        state,
        evaluator,
        msg.simulations,
        1.5,
        () => msg.requestId !== latestRequestId,
      );

      if (msg.requestId !== latestRequestId) return;
      self.postMessage({
        type: 'result',
        requestId: msg.requestId,
        tier: msg.tier,
        analysis,
      });
    } catch (err) {
      self.postMessage({
        type: 'error',
        requestId: msg.requestId,
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }
};
