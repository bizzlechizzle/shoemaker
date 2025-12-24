/**
 * Progress Reporter Tests
 *
 * Tests for Unix socket-based progress reporting.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createServer, Server, Socket } from 'net';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  ProgressReporter,
  getProgressReporter,
  SHOEMAKER_STAGES,
  type ShoemakerStage,
} from '../src/core/progress-reporter.js';

describe('ProgressReporter', () => {
  let reporter: ProgressReporter;
  let server: Server;
  let socketPath: string;
  let tempDir: string;
  let serverConnection: Socket | null = null;
  let receivedMessages: string[] = [];

  beforeEach(() => {
    // Create temp directory for socket
    tempDir = mkdtempSync(join(tmpdir(), 'shoe-test-'));
    socketPath = join(tempDir, 'progress.sock');

    // Create mock server
    receivedMessages = [];
    serverConnection = null;

    server = createServer((socket) => {
      serverConnection = socket;
      socket.on('data', (data) => {
        receivedMessages.push(...data.toString().split('\n').filter(Boolean));
      });
    });

    server.listen(socketPath);

    // Set environment variable
    process.env.PROGRESS_SOCKET = socketPath;
    process.env.PROGRESS_SESSION_ID = 'test-session-456';

    reporter = new ProgressReporter();
  });

  afterEach(async () => {
    reporter.close();
    server.close();

    // Clean up temp directory
    try {
      rmSync(tempDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }

    delete process.env.PROGRESS_SOCKET;
    delete process.env.PROGRESS_SESSION_ID;
  });

  describe('SHOEMAKER_STAGES', () => {
    it('should have correct stage weights summing to 100', () => {
      const totalWeight = Object.values(SHOEMAKER_STAGES).reduce((sum, s) => sum + s.weight, 0);
      expect(totalWeight).toBe(100);
    });

    it('should have all required stages', () => {
      expect(SHOEMAKER_STAGES.scanning).toBeDefined();
      expect(SHOEMAKER_STAGES.analyzing).toBeDefined();
      expect(SHOEMAKER_STAGES.generating).toBeDefined();
      expect(SHOEMAKER_STAGES['writing-xmp']).toBeDefined();
    });

    it('should have sequential stage numbers', () => {
      const stages = Object.values(SHOEMAKER_STAGES);
      stages.forEach((stage, index) => {
        expect(stage.number).toBe(index + 1);
        expect(stage.totalStages).toBe(4);
      });
    });
  });

  describe('connect()', () => {
    it('should connect to socket when PROGRESS_SOCKET is set', async () => {
      const connected = await reporter.connect();

      expect(connected).toBe(true);
      expect(reporter.isConnected).toBe(true);
    });

    it('should return false when PROGRESS_SOCKET is not set', async () => {
      delete process.env.PROGRESS_SOCKET;
      reporter = new ProgressReporter();

      const connected = await reporter.connect();

      expect(connected).toBe(false);
      expect(reporter.isConnected).toBe(false);
    });
  });

  describe('stageStarted()', () => {
    it('should send stage_started message for scanning stage', async () => {
      await reporter.connect();

      reporter.stageStarted('scanning');

      await new Promise((r) => setTimeout(r, 50));

      const msg = JSON.parse(receivedMessages[0]);

      expect(msg.type).toBe('stage_started');
      expect(msg.stage.name).toBe('scanning');
      expect(msg.stage.display_name).toBe('Scanning files');
      expect(msg.stage.number).toBe(1);
      expect(msg.stage.total_stages).toBe(4);
      expect(msg.app).toBe('shoemaker');
    });

    it('should send stage_started message for generating stage', async () => {
      await reporter.connect();

      reporter.stageStarted('generating');

      await new Promise((r) => setTimeout(r, 50));

      const msg = JSON.parse(receivedMessages[0]);

      expect(msg.stage.name).toBe('generating');
      expect(msg.stage.display_name).toBe('Generating thumbnails');
      expect(msg.stage.number).toBe(3);
    });
  });

  describe('stageCompleted()', () => {
    it('should send stage_completed message', async () => {
      await reporter.connect();

      reporter.stageCompleted('scanning', 1500, 50);

      await new Promise((r) => setTimeout(r, 50));

      const msg = JSON.parse(receivedMessages[0]);

      expect(msg.type).toBe('stage_completed');
      expect(msg.stage.name).toBe('scanning');
      expect(msg.stage.number).toBe(1);
      expect(msg.duration_ms).toBe(1500);
      expect(msg.items_processed).toBe(50);
    });
  });

  describe('progress()', () => {
    it('should send progress message with stage info', async () => {
      await reporter.connect();
      reporter.resetStartTime();

      reporter.progress('generating', {
        completed: 25,
        total: 100,
        failed: 1,
        skipped: 2,
        currentFile: '/path/to/image.jpg',
        percentComplete: 25.0,
        etaMs: 45000,
      });

      await new Promise((r) => setTimeout(r, 50));

      const msg = JSON.parse(receivedMessages[0]);

      expect(msg.type).toBe('progress');
      expect(msg.stage.name).toBe('generating');
      expect(msg.stage.display_name).toBe('Generating thumbnails');
      expect(msg.stage.weight).toBe(80);
      expect(msg.items.completed).toBe(25);
      expect(msg.items.total).toBe(100);
      expect(msg.items.failed).toBe(1);
      expect(msg.current.item).toBe('/path/to/image.jpg');
      expect(msg.current.item_short).toBe('image.jpg');
      expect(msg.timing.eta_ms).toBe(45000);
      expect(msg.percent_complete).toBe(25.0);
    });
  });

  describe('complete()', () => {
    it('should send complete message with summary', async () => {
      await reporter.connect();

      reporter.complete({
        totalItems: 100,
        successful: 98,
        failed: 1,
        skipped: 1,
        durationMs: 30000,
      });

      await new Promise((r) => setTimeout(r, 50));

      const msg = JSON.parse(receivedMessages[0]);

      expect(msg.type).toBe('complete');
      expect(msg.summary.total_items).toBe(100);
      expect(msg.summary.successful).toBe(98);
      expect(msg.summary.failed).toBe(1);
      expect(msg.summary.skipped).toBe(1);
      expect(msg.summary.duration_ms).toBe(30000);
      expect(msg.exit_code).toBe(1);
    });
  });

  describe('control commands', () => {
    it('should handle pause/resume cycle', async () => {
      await reporter.connect();

      // Pause
      serverConnection?.write(JSON.stringify({ type: 'control', command: 'pause' }) + '\n');
      await new Promise((r) => setTimeout(r, 50));

      expect(reporter.paused).toBe(true);
      expect(reporter.shouldContinue()).toBe(true); // Not cancelled

      // Resume
      serverConnection?.write(JSON.stringify({ type: 'control', command: 'resume' }) + '\n');
      await new Promise((r) => setTimeout(r, 50));

      expect(reporter.paused).toBe(false);
    });

    it('should handle cancel command', async () => {
      await reporter.connect();

      serverConnection?.write(
        JSON.stringify({ type: 'control', command: 'cancel', reason: 'Timeout' }) + '\n'
      );
      await new Promise((r) => setTimeout(r, 50));

      expect(reporter.cancelled).toBe(true);
      expect(reporter.shouldContinue()).toBe(false);
    });
  });

  describe('waitWhilePaused()', () => {
    it('should block while paused', async () => {
      await reporter.connect();

      serverConnection?.write(JSON.stringify({ type: 'control', command: 'pause' }) + '\n');
      await new Promise((r) => setTimeout(r, 50));

      let completed = false;
      const waitPromise = reporter.waitWhilePaused().then(() => {
        completed = true;
      });

      await new Promise((r) => setTimeout(r, 150));
      expect(completed).toBe(false);

      serverConnection?.write(JSON.stringify({ type: 'control', command: 'resume' }) + '\n');
      await waitPromise;
      expect(completed).toBe(true);
    });
  });
});

describe('Standalone mode', () => {
  it('should operate silently without socket', () => {
    delete process.env.PROGRESS_SOCKET;

    const reporter = new ProgressReporter();

    expect(() => {
      reporter.stageStarted('scanning');
      reporter.progress('generating', {
        completed: 50,
        total: 100,
        percentComplete: 50,
      });
      reporter.complete({
        totalItems: 100,
        successful: 100,
        failed: 0,
        skipped: 0,
        durationMs: 5000,
      });
    }).not.toThrow();

    expect(reporter.isConnected).toBe(false);
  });
});
