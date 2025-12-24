/**
 * CLI Progress Tracking tests
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  ProgressTracker,
  formatDuration,
  formatProgressBar,
  formatProgressLine,
  formatSummary,
  type ProgressState,
} from '../../src/cli/progress.js';

describe('formatDuration', () => {
  it('should format sub-second durations', () => {
    expect(formatDuration(0)).toBe('< 1s');
    expect(formatDuration(500)).toBe('< 1s');
    expect(formatDuration(999)).toBe('< 1s');
  });

  it('should format seconds', () => {
    expect(formatDuration(1000)).toBe('1s');
    expect(formatDuration(5000)).toBe('5s');
    expect(formatDuration(45000)).toBe('45s');
    expect(formatDuration(59000)).toBe('59s');
  });

  it('should format minutes and seconds', () => {
    expect(formatDuration(60000)).toBe('1m');
    expect(formatDuration(61000)).toBe('1m1s');
    expect(formatDuration(90000)).toBe('1m30s');
    expect(formatDuration(150000)).toBe('2m30s');
    expect(formatDuration(3599000)).toBe('59m59s');
  });

  it('should format hours and minutes', () => {
    expect(formatDuration(3600000)).toBe('1h');
    expect(formatDuration(3660000)).toBe('1h1m');
    expect(formatDuration(5400000)).toBe('1h30m');
    expect(formatDuration(7200000)).toBe('2h');
    expect(formatDuration(7260000)).toBe('2h1m');
  });

  it('should handle edge cases', () => {
    expect(formatDuration(-1000)).toBe('--');
    expect(formatDuration(Infinity)).toBe('--');
    expect(formatDuration(NaN)).toBe('--');
  });
});

describe('formatProgressBar', () => {
  it('should render empty bar at 0%', () => {
    expect(formatProgressBar(0, 10)).toBe('[░░░░░░░░░░]');
  });

  it('should render full bar at 100%', () => {
    expect(formatProgressBar(100, 10)).toBe('[██████████]');
  });

  it('should render partial bar at 50%', () => {
    expect(formatProgressBar(50, 10)).toBe('[█████░░░░░]');
  });

  it('should render with default width', () => {
    const bar = formatProgressBar(50);
    expect(bar).toBe('[██████████░░░░░░░░░░]');
    expect(bar.length).toBe(22); // 20 chars + 2 brackets
  });

  it('should clamp values below 0', () => {
    expect(formatProgressBar(-10, 10)).toBe('[░░░░░░░░░░]');
  });

  it('should clamp values above 100', () => {
    expect(formatProgressBar(150, 10)).toBe('[██████████]');
  });

  it('should handle various percentages correctly', () => {
    expect(formatProgressBar(25, 20)).toBe('[█████░░░░░░░░░░░░░░░]');
    expect(formatProgressBar(75, 20)).toBe('[███████████████░░░░░]');
    expect(formatProgressBar(10, 10)).toBe('[█░░░░░░░░░]');
    expect(formatProgressBar(90, 10)).toBe('[█████████░]');
  });
});

describe('formatProgressLine', () => {
  it('should format a complete progress line', () => {
    const state: ProgressState = {
      completed: 50,
      total: 100,
      current: 'test_file.jpg',
      status: 'processing',
      startTime: Date.now() - 10000,
      elapsedMs: 10000,
      throughput: 5.0,
      etaMs: 10000,
      percent: 50,
    };

    const line = formatProgressLine(state);

    expect(line).toContain('[██████████░░░░░░░░░░]');
    expect(line).toContain('50%');
    expect(line).toContain('50/100 files');
    expect(line).toContain('5.0 files/s');
    expect(line).toContain('ETA: 10s');
    expect(line).toContain('test_file.jpg');
  });

  it('should show -- for zero throughput', () => {
    const state: ProgressState = {
      completed: 0,
      total: 100,
      current: 'file.jpg',
      status: 'processing',
      startTime: Date.now(),
      elapsedMs: 0,
      throughput: 0,
      etaMs: 0,
      percent: 0,
    };

    const line = formatProgressLine(state);
    expect(line).toContain('--');
    expect(line).toContain('ETA: --');
  });

  it('should truncate long filenames', () => {
    const state: ProgressState = {
      completed: 1,
      total: 10,
      current: 'very_long_filename_that_should_be_truncated.jpg',
      status: 'success',
      startTime: Date.now(),
      elapsedMs: 1000,
      throughput: 1.0,
      etaMs: 9000,
      percent: 10,
    };

    const line = formatProgressLine(state);
    // Should be truncated to ~30 chars
    expect(line).toContain('...');
    expect(line.length).toBeLessThan(120);
  });
});

describe('formatSummary', () => {
  it('should format a successful batch summary', () => {
    const summary = formatSummary(100, 95, 3, 2, 60000);

    expect(summary).toContain('Completed: 100 files in 1m');
    expect(summary).toContain('95 succeeded');
    expect(summary).toContain('2 skipped');
    expect(summary).toContain('3 failed');
    expect(summary).toContain('Average: 600ms per file');
  });

  it('should handle all succeeded', () => {
    const summary = formatSummary(50, 50, 0, 0, 25000);

    expect(summary).toContain('50 succeeded');
    expect(summary).not.toContain('skipped');
    expect(summary).not.toContain('failed');
  });

  it('should handle zero duration', () => {
    const summary = formatSummary(0, 0, 0, 0, 0);

    expect(summary).toContain('Completed: 0 files');
    expect(summary).not.toContain('Average:');
  });
});

describe('ProgressTracker', () => {
  beforeEach(() => {
    // Mock process.stdout.isTTY
    vi.stubGlobal('process', {
      ...process,
      stdout: {
        ...process.stdout,
        isTTY: true,
        write: vi.fn(),
      },
    });
  });

  it('should initialize with correct values', () => {
    const tracker = new ProgressTracker(100);
    const state = tracker.getState();

    expect(state.total).toBe(100);
    expect(state.completed).toBe(0);
    expect(state.percent).toBe(0);
    expect(state.throughput).toBe(0);
  });

  it('should update state correctly', () => {
    const tracker = new ProgressTracker(100);

    tracker.update({
      current: 'file1.jpg',
      completed: 10,
      total: 100,
      status: 'success',
      duration: 500,
    });

    const state = tracker.getState();

    expect(state.completed).toBe(10);
    expect(state.current).toBe('file1.jpg');
    expect(state.status).toBe('success');
    expect(state.percent).toBe(10);
  });

  it('should calculate throughput from duration', () => {
    const tracker = new ProgressTracker(100);

    // First update with 500ms duration = 2 files/sec
    tracker.update({
      current: 'file1.jpg',
      completed: 1,
      total: 100,
      status: 'success',
      duration: 500,
    });

    const state = tracker.getState();
    expect(state.throughput).toBe(2); // 1000ms / 500ms = 2 files/sec
  });

  it('should apply EWMA smoothing to throughput', () => {
    const tracker = new ProgressTracker(100);

    // First: 500ms = 2 files/sec
    tracker.update({
      current: 'file1.jpg',
      completed: 1,
      total: 100,
      status: 'success',
      duration: 500,
    });

    const state1 = tracker.getState();
    expect(state1.throughput).toBe(2);

    // Second: 1000ms = 1 file/sec
    // EWMA: 0.15 * 1 + 0.85 * 2 = 0.15 + 1.7 = 1.85
    tracker.update({
      current: 'file2.jpg',
      completed: 2,
      total: 100,
      status: 'success',
      duration: 1000,
    });

    const state2 = tracker.getState();
    expect(state2.throughput).toBeCloseTo(1.85, 2);
  });

  it('should calculate ETA from throughput', () => {
    const tracker = new ProgressTracker(100);

    tracker.update({
      current: 'file1.jpg',
      completed: 50,
      total: 100,
      status: 'success',
      duration: 1000, // 1 file/sec
    });

    const state = tracker.getState();
    // 50 remaining files at 1 file/sec = 50 seconds = 50000ms
    expect(state.etaMs).toBeCloseTo(50000, -2);
  });

  it('should track elapsed time', () => {
    const tracker = new ProgressTracker(100);

    // Wait a bit
    const start = Date.now();
    while (Date.now() - start < 50) {
      // busy wait
    }

    const elapsed = tracker.getElapsedMs();
    expect(elapsed).toBeGreaterThanOrEqual(50);
  });

  it('should report TTY status', () => {
    const tracker = new ProgressTracker(100);
    expect(tracker.isInteractive()).toBe(true);
  });

  it('should handle processing status (no throughput update)', () => {
    const tracker = new ProgressTracker(100);

    tracker.update({
      current: 'file1.jpg',
      completed: 0,
      total: 100,
      status: 'processing',
    });

    const state = tracker.getState();
    expect(state.throughput).toBe(0);
    expect(state.status).toBe('processing');
  });

  it('should handle skipped status', () => {
    const tracker = new ProgressTracker(100);

    tracker.update({
      current: 'file1.jpg',
      completed: 1,
      total: 100,
      status: 'skipped',
      duration: 10,
    });

    const state = tracker.getState();
    expect(state.status).toBe('skipped');
    // Should still calculate throughput for skipped files
    expect(state.throughput).toBeGreaterThan(0);
  });

  it('should handle error status', () => {
    const tracker = new ProgressTracker(100);

    tracker.update({
      current: 'file1.jpg',
      completed: 1,
      total: 100,
      status: 'error',
      message: 'File not found',
      duration: 100,
    });

    const state = tracker.getState();
    expect(state.status).toBe('error');
  });
});
