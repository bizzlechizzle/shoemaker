/**
 * Progress Reporter - Reports progress to orchestrator via Unix socket
 *
 * Enables bidirectional communication:
 * - Worker sends progress updates, stage changes, completion
 * - Orchestrator sends control commands: pause, resume, cancel
 *
 * Falls back to standalone mode (stderr output) if no socket configured.
 */

import { createConnection, Socket } from 'net';
import { createInterface, Interface } from 'readline';
import { EventEmitter } from 'events';

// App identification
const APP_NAME = 'shoemaker';
let APP_VERSION = '0.1.10';
try {
  APP_VERSION = process.env.npm_package_version || '0.1.10';
} catch {
  // Use default
}

// Stage definitions for shoemaker
export const SHOEMAKER_STAGES = {
  scanning: { name: 'scanning', displayName: 'Scanning files', number: 1, totalStages: 4, weight: 5 },
  analyzing: { name: 'analyzing', displayName: 'Analyzing previews', number: 2, totalStages: 4, weight: 10 },
  generating: { name: 'generating', displayName: 'Generating thumbnails', number: 3, totalStages: 4, weight: 80 },
  'writing-xmp': { name: 'writing-xmp', displayName: 'Writing XMP', number: 4, totalStages: 4, weight: 5 },
} as const;

export type ShoemakerStage = keyof typeof SHOEMAKER_STAGES;

export interface ProgressMessage {
  type: string;
  timestamp: string;
  session_id: string;
  app: string;
  app_version: string;
  [key: string]: unknown;
}

export interface StageInfo {
  name: string;
  displayName: string;
  number: number;
  totalStages: number;
  weight: number;
}

export interface ProgressData {
  stage: StageInfo;
  completed: number;
  total: number;
  failed?: number;
  skipped?: number;
  currentFile?: string;
  percentComplete: number;
  etaMs?: number;
}

/**
 * Progress Reporter for shoemaker
 */
export class ProgressReporter extends EventEmitter {
  private socket: Socket | null = null;
  private rl: Interface | null = null;
  private _paused = false;
  private _cancelled = false;
  private sessionId: string;
  private connected = false;
  private startedAt: number;

  constructor() {
    super();
    this.sessionId = process.env.PROGRESS_SESSION_ID || '';
    this.startedAt = Date.now();
  }

  get paused(): boolean {
    return this._paused;
  }

  get cancelled(): boolean {
    return this._cancelled;
  }

  get isConnected(): boolean {
    return this.connected;
  }

  /**
   * Connect to orchestrator socket
   */
  async connect(): Promise<boolean> {
    const socketPath = process.env.PROGRESS_SOCKET;
    if (!socketPath) {
      return false;
    }

    return new Promise((resolve) => {
      try {
        this.socket = createConnection(socketPath, () => {
          this.connected = true;
          this.setupListener();
          resolve(true);
        });
        this.socket.on('error', () => {
          this.connected = false;
          resolve(false);
        });
        this.socket.on('close', () => {
          this.connected = false;
        });
      } catch {
        resolve(false);
      }
    });
  }

  private setupListener(): void {
    if (!this.socket) return;

    this.rl = createInterface({ input: this.socket });
    this.rl.on('line', (line) => {
      try {
        const msg = JSON.parse(line);
        if (msg.type === 'control') {
          this.handleControl(msg);
        }
      } catch {
        // Ignore malformed messages
      }
    });
  }

  private handleControl(msg: { command: string; reason?: string }): void {
    switch (msg.command) {
      case 'pause':
        this._paused = true;
        this.sendAck('pause', 'accepted');
        this.emit('pause');
        break;
      case 'resume':
        this._paused = false;
        this.sendAck('resume', 'accepted');
        this.emit('resume');
        break;
      case 'cancel':
        this._cancelled = true;
        this.sendAck('cancel', 'accepted');
        this.emit('cancel', msg.reason);
        break;
    }
  }

  private sendAck(command: string, status: string): void {
    this.send({ type: 'ack', command, status });
  }

  send(message: Partial<ProgressMessage>): void {
    if (!this.socket || !this.connected) return;

    const fullMessage: ProgressMessage = {
      type: message.type || 'progress',
      timestamp: new Date().toISOString(),
      session_id: this.sessionId,
      app: APP_NAME,
      app_version: APP_VERSION,
      ...message,
    };

    try {
      this.socket.write(JSON.stringify(fullMessage) + '\n');
    } catch {
      // Socket may have closed
    }
  }

  stageStarted(stageName: ShoemakerStage): void {
    const stage = SHOEMAKER_STAGES[stageName];
    this.send({
      type: 'stage_started',
      stage: {
        name: stage.name,
        display_name: stage.displayName,
        number: stage.number,
        total_stages: stage.totalStages,
      },
    });
  }

  stageCompleted(stageName: ShoemakerStage, durationMs: number, itemsProcessed: number): void {
    const stage = SHOEMAKER_STAGES[stageName];
    this.send({
      type: 'stage_completed',
      stage: { name: stage.name, number: stage.number },
      duration_ms: durationMs,
      items_processed: itemsProcessed,
    });
  }

  progress(stageName: ShoemakerStage, data: Omit<ProgressData, 'stage'>): void {
    const stage = SHOEMAKER_STAGES[stageName];
    this.send({
      type: 'progress',
      stage: {
        name: stage.name,
        display_name: stage.displayName,
        number: stage.number,
        total_stages: stage.totalStages,
        weight: stage.weight,
      },
      items: {
        total: data.total,
        completed: data.completed,
        failed: data.failed || 0,
        skipped: data.skipped || 0,
      },
      current: {
        item: data.currentFile,
        item_short: data.currentFile?.split('/').pop(),
      },
      timing: {
        started_at: new Date(this.startedAt).toISOString(),
        elapsed_ms: Date.now() - this.startedAt,
        eta_ms: data.etaMs,
      },
      percent_complete: data.percentComplete,
    });
  }

  complete(summary: {
    totalItems: number;
    successful: number;
    failed: number;
    skipped: number;
    durationMs: number;
  }): void {
    this.send({
      type: 'complete',
      summary: {
        total_items: summary.totalItems,
        successful: summary.successful,
        failed: summary.failed,
        skipped: summary.skipped,
        duration_ms: summary.durationMs,
      },
      exit_code: summary.failed > 0 ? 1 : 0,
    });
  }

  async waitWhilePaused(): Promise<void> {
    while (this._paused && !this._cancelled) {
      await new Promise((r) => setTimeout(r, 100));
    }
  }

  shouldContinue(): boolean {
    return !this._cancelled;
  }

  resetStartTime(): void {
    this.startedAt = Date.now();
  }

  close(): void {
    this.rl?.close();
    this.socket?.end();
    this.connected = false;
  }
}

let _reporter: ProgressReporter | null = null;

export function getProgressReporter(): ProgressReporter {
  if (!_reporter) {
    _reporter = new ProgressReporter();
  }
  return _reporter;
}
