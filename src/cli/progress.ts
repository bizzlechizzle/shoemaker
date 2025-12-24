/**
 * CLI Progress Tracking with EWMA-smoothed ETA estimation
 *
 * Provides a progress bar with throughput and time remaining:
 * [████████░░░░░░░░░░░░] 45% | 234/520 files | 12.3 files/s | ETA: 2m30s | photo_001.jpg
 */

import type { ProgressInfo } from '../services/thumbnail-generator.js';

export interface ProgressState {
  completed: number;
  total: number;
  current: string;
  status: 'processing' | 'success' | 'error' | 'skipped';
  startTime: number;
  elapsedMs: number;
  throughput: number;      // EWMA-smoothed files/sec
  etaMs: number;           // Estimated time remaining in ms
  percent: number;
}

export class ProgressTracker {
  private startTime: number;
  private throughput: number = 0;
  private completed: number = 0;
  private total: number;
  private current: string = '';
  private status: ProgressInfo['status'] = 'processing';
  private isTTY: boolean;
  private lastRenderTime: number = 0;
  private readonly minRenderInterval: number = 100; // ms between renders
  private readonly ewmaAlpha: number = 0.15;

  constructor(total: number) {
    this.total = total;
    this.startTime = Date.now();
    this.isTTY = process.stdout.isTTY ?? false;
  }

  /**
   * Update progress state from a ProgressInfo callback
   */
  update(info: ProgressInfo): void {
    const now = Date.now();

    this.completed = info.completed;
    this.current = info.current;
    this.status = info.status;
    this.total = info.total;

    // Calculate instantaneous throughput from file duration if available
    if (info.duration && info.duration > 0 && info.status !== 'processing') {
      const instantThroughput = 1000 / info.duration; // files per second

      // EWMA smoothing
      if (this.throughput === 0) {
        this.throughput = instantThroughput;
      } else {
        this.throughput = this.ewmaAlpha * instantThroughput + (1 - this.ewmaAlpha) * this.throughput;
      }
    } else if (this.completed > 0) {
      // Fallback: calculate from overall elapsed time
      const elapsed = now - this.startTime;
      if (elapsed > 0) {
        const instantThroughput = (this.completed * 1000) / elapsed;
        if (this.throughput === 0) {
          this.throughput = instantThroughput;
        } else {
          this.throughput = this.ewmaAlpha * instantThroughput + (1 - this.ewmaAlpha) * this.throughput;
        }
      }
    }
  }

  /**
   * Get current progress state
   */
  getState(): ProgressState {
    const now = Date.now();
    const elapsedMs = now - this.startTime;
    const remaining = this.total - this.completed;
    const etaMs = this.throughput > 0 ? (remaining / this.throughput) * 1000 : 0;
    const percent = this.total > 0 ? (this.completed / this.total) * 100 : 0;

    return {
      completed: this.completed,
      total: this.total,
      current: this.current,
      status: this.status,
      startTime: this.startTime,
      elapsedMs,
      throughput: this.throughput,
      etaMs,
      percent,
    };
  }

  /**
   * Render progress bar to stdout (respects TTY and rate limiting)
   */
  render(): void {
    if (!this.isTTY) return;

    const now = Date.now();
    if (now - this.lastRenderTime < this.minRenderInterval) return;
    this.lastRenderTime = now;

    const state = this.getState();
    const line = formatProgressLine(state);

    // Clear line and write new content
    process.stdout.write('\r\x1b[K' + line);
  }

  /**
   * Force render regardless of rate limiting
   */
  forceRender(): void {
    if (!this.isTTY) return;

    const state = this.getState();
    const line = formatProgressLine(state);
    process.stdout.write('\r\x1b[K' + line);
  }

  /**
   * Clear the progress line
   */
  clear(): void {
    if (!this.isTTY) return;
    process.stdout.write('\r\x1b[K');
  }

  /**
   * Get elapsed time since start
   */
  getElapsedMs(): number {
    return Date.now() - this.startTime;
  }

  /**
   * Check if running in a TTY
   */
  isInteractive(): boolean {
    return this.isTTY;
  }
}

/**
 * Format duration in human-readable form
 * Examples: "< 1s", "45s", "2m30s", "1h15m"
 */
export function formatDuration(ms: number): string {
  if (!isFinite(ms) || ms < 0) return '--';
  if (ms < 1000) return '< 1s';

  const seconds = Math.floor(ms / 1000) % 60;
  const minutes = Math.floor(ms / 60000) % 60;
  const hours = Math.floor(ms / 3600000);

  if (hours > 0) {
    return `${hours}h${minutes > 0 ? minutes + 'm' : ''}`;
  }
  if (minutes > 0) {
    return `${minutes}m${seconds > 0 ? seconds + 's' : ''}`;
  }
  return `${seconds}s`;
}

/**
 * Format a progress bar with Unicode block characters
 * Example: [████████░░░░░░░░░░░░]
 */
export function formatProgressBar(percent: number, width: number = 20): string {
  const clamped = Math.max(0, Math.min(100, percent));
  const filled = Math.round((clamped / 100) * width);
  const empty = width - filled;
  return `[${'█'.repeat(filled)}${'░'.repeat(empty)}]`;
}

/**
 * Format the full progress line for CLI display
 * Example: [████████░░░░░░░░░░░░] 45% | 234/520 files | 12.3 files/s | ETA: 2m30s | photo_001.jpg
 */
export function formatProgressLine(state: ProgressState): string {
  const bar = formatProgressBar(state.percent, 20);
  const pct = state.percent.toFixed(0).padStart(3);
  const count = `${state.completed}/${state.total}`;
  const throughput = state.throughput > 0 ? `${state.throughput.toFixed(1)} files/s` : '--';
  const eta = state.etaMs > 0 ? `ETA: ${formatDuration(state.etaMs)}` : 'ETA: --';

  // Truncate filename if too long
  const maxFileLen = 30;
  let filename = state.current ? truncateMiddle(state.current, maxFileLen) : '';

  // Status icon
  const icon = getStatusIcon(state.status);

  return `${bar} ${pct}% | ${count} files | ${throughput} | ${eta} | ${icon} ${filename}`;
}

/**
 * Get status icon for current state
 */
function getStatusIcon(status: ProgressInfo['status']): string {
  switch (status) {
    case 'success': return '\x1b[32m✓\x1b[0m';  // green
    case 'error': return '\x1b[31m✗\x1b[0m';    // red
    case 'skipped': return '\x1b[33m⏭\x1b[0m';  // yellow
    case 'processing': return '\x1b[36m→\x1b[0m'; // cyan
    default: return '→';
  }
}

/**
 * Truncate a string in the middle, preserving start and end
 * Example: "very_long_filename.jpg" -> "very_lon...me.jpg"
 */
function truncateMiddle(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str;

  const ellipsis = '...';
  const charsToShow = maxLen - ellipsis.length;
  const frontChars = Math.ceil(charsToShow / 2);
  const backChars = Math.floor(charsToShow / 2);

  return str.slice(0, frontChars) + ellipsis + str.slice(-backChars);
}

/**
 * Format a summary line after batch completion
 */
export function formatSummary(
  total: number,
  succeeded: number,
  failed: number,
  skipped: number,
  durationMs: number
): string {
  const lines: string[] = [];

  lines.push(`\nCompleted: ${total} files in ${formatDuration(durationMs)}`);

  const parts: string[] = [];
  if (succeeded > 0) parts.push(`\x1b[32m${succeeded} succeeded\x1b[0m`);
  if (skipped > 0) parts.push(`\x1b[33m${skipped} skipped\x1b[0m`);
  if (failed > 0) parts.push(`\x1b[31m${failed} failed\x1b[0m`);

  if (parts.length > 0) {
    lines.push(`  ${parts.join(' | ')}`);
  }

  if (total > 0 && durationMs > 0) {
    const avgMs = durationMs / total;
    lines.push(`  Average: ${avgMs.toFixed(0)}ms per file`);
  }

  return lines.join('\n');
}
