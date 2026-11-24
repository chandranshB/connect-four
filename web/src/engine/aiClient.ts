import type { GameState } from './c4';
import type { ModelTier } from './ai.worker';
import type { SearchAnalysis } from './mcts';

export type { ModelTier, SearchAnalysis };

export interface TierBenchmark {
  iteration: number;
  raw_win1_acc: number;
  raw_block1_acc: number;
  raw_vs_mm2_pct: number;
  mcts64_vs_mm2_pct: number;
  mcts96_vs_mm4_pct: number;
}

export interface ModelInfoReport {
  device: string;
  params: number;
  elapsed_seconds: number;
  total_selfplay_games: number;
  total_augmented_samples: number;
  tiers: Record<ModelTier, TierBenchmark>;
  exported: Record<
    ModelTier,
    {
      onnx_file: string;
      iteration: number;
      params: number;
      size_kb: number;
      max_logits_diff: number;
      max_value_diff: number;
    }
  >;
}

type WorkerOutMsg =
  | { type: 'ready'; tier: ModelTier }
  | { type: 'result'; requestId: number; tier: ModelTier; analysis: SearchAnalysis }
  | { type: 'error'; requestId: number; message: string };

export class Connect4AIClient {
  private worker: Worker;
  private nextRequestId = 1;
  private pending = new Map<
    number,
    { resolve: (res: SearchAnalysis) => void; reject: (err: Error) => void }
  >();

  constructor() {
    this.worker = new Worker(new URL('./ai.worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.onmessage = (ev: MessageEvent<WorkerOutMsg>) => {
      const msg = ev.data;
      if (msg.type === 'result') {
        const entry = this.pending.get(msg.requestId);
        if (entry) {
          this.pending.delete(msg.requestId);
          entry.resolve(msg.analysis);
        }
      } else if (msg.type === 'error') {
        if (msg.requestId >= 0) {
          const entry = this.pending.get(msg.requestId);
          if (entry) {
            this.pending.delete(msg.requestId);
            entry.reject(new Error(msg.message));
          }
        }
      }
    };
  }

  preload(tier: ModelTier): void {
    this.worker.postMessage({ type: 'preload', tier });
  }

  analyze(state: GameState, tier: ModelTier, simulations: number): Promise<SearchAnalysis> {
    // Supersede any older in-flight request
    for (const [id, entry] of this.pending.entries()) {
      this.pending.delete(id);
      entry.reject(new DOMException('Superseded by newer search', 'AbortError'));
    }

    const requestId = this.nextRequestId++;
    return new Promise<SearchAnalysis>((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      this.worker.postMessage({
        type: 'analyze',
        requestId,
        tier,
        simulations,
        board: Array.from(state.board),
        ply: state.ply,
        currentPlayer: state.currentPlayer,
        winner: state.winner,
        isDraw: state.isDraw,
      });
    });
  }

  dispose(): void {
    for (const entry of this.pending.values()) {
      entry.reject(new DOMException('Worker disposed', 'AbortError'));
    }
    this.pending.clear();
    this.worker.terminate();
  }
}
