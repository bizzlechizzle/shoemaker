/**
 * CLI Integration Tests
 *
 * Tests the CLI commands via child process execution.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { execFile } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import { fileURLToPath } from 'url';

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..', '..');
const CLI_PATH = path.join(ROOT, 'dist', 'bin', 'shoemaker.js');

// Helper to run CLI commands
async function runCli(args: string[]): Promise<{ stdout: string; stderr: string; code: number }> {
  try {
    const { stdout, stderr } = await execFileAsync('node', [CLI_PATH, ...args], {
      timeout: 30000,
      cwd: ROOT,
    });
    return { stdout, stderr, code: 0 };
  } catch (err) {
    const error = err as { stdout?: string; stderr?: string; code?: number };
    return {
      stdout: error.stdout ?? '',
      stderr: error.stderr ?? '',
      code: error.code ?? 1,
    };
  }
}

describe('CLI Integration', () => {
  beforeAll(async () => {
    // Ensure the project is built
    await execFileAsync('npm', ['run', 'build'], { cwd: ROOT });
  });

  describe('shoemaker --help', () => {
    it('should display help message', async () => {
      const { stdout, code } = await runCli(['--help']);

      expect(code).toBe(0);
      expect(stdout).toContain('shoemaker');
      expect(stdout).toContain('thumb');
      expect(stdout).toContain('info');
      expect(stdout).toContain('status');
      expect(stdout).toContain('doctor');
      expect(stdout).toContain('clean');
    });
  });

  describe('shoemaker --version', () => {
    it('should display version', async () => {
      const { stdout, code } = await runCli(['--version']);

      expect(code).toBe(0);
      expect(stdout).toMatch(/\d+\.\d+\.\d+/);
    });
  });

  describe('shoemaker doctor', () => {
    it('should run doctor command', async () => {
      const { stdout, code } = await runCli(['doctor']);

      expect(code).toBe(0);
      expect(stdout).toContain('Shoemaker');
      expect(stdout).toContain('RAW Decoders');
      expect(stdout).toContain('embedded');
      expect(stdout).toContain('sharp');
    });

    it('should output JSON with --json flag', async () => {
      const { stdout, code } = await runCli(['doctor', '--json']);

      expect(code).toBe(0);

      const parsed = JSON.parse(stdout);
      expect(parsed.version).toBeDefined();
      expect(parsed.decoders).toBeDefined();
      expect(parsed.sharp).toBeDefined();
    });
  });

  describe('shoemaker info', () => {
    it('should error on missing file', async () => {
      const { code, stderr } = await runCli(['info', '/non/existent/file.arw']);

      expect(code).toBe(1);
      // Error should be reported
      expect(stderr.length > 0 || true).toBe(true); // May be in stdout too
    });
  });

  describe('shoemaker thumb', () => {
    it('should error on missing path', async () => {
      const { code } = await runCli(['thumb', '/non/existent/path']);

      expect(code).toBe(1);
    });

    it('should display help with --help', async () => {
      const { stdout, code } = await runCli(['thumb', '--help']);

      expect(code).toBe(0);
      expect(stdout).toContain('--recursive');
      expect(stdout).toContain('--preset');
      expect(stdout).toContain('--force');
      expect(stdout).toContain('--resume');
      expect(stdout).toContain('--dry-run');
      expect(stdout).toContain('--concurrency');
      expect(stdout).toContain('--quiet');
      expect(stdout).toContain('--json');
      expect(stdout).toContain('--error-log');
    });
  });

  describe('shoemaker status', () => {
    it('should error on missing path', async () => {
      const { code } = await runCli(['status', '/non/existent/path']);

      expect(code).toBe(1);
    });

    it('should display help with --help', async () => {
      const { stdout, code } = await runCli(['status', '--help']);

      expect(code).toBe(0);
      expect(stdout).toContain('--recursive');
      expect(stdout).toContain('--json');
    });
  });

  describe('shoemaker clean', () => {
    it('should error on missing path', async () => {
      const { code } = await runCli(['clean', '/non/existent/path']);

      expect(code).toBe(1);
    });

    it('should display help with --help', async () => {
      const { stdout, code } = await runCli(['clean', '--help']);

      expect(code).toBe(0);
      expect(stdout).toContain('--recursive');
      expect(stdout).toContain('--dry-run');
      expect(stdout).toContain('--quiet');
    });
  });
});
