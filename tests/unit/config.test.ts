/**
 * Config tests
 */

import { describe, it, expect } from 'vitest';
import { expandPath, getConfigPaths } from '../../src/core/config.js';
import os from 'os';
import path from 'path';

describe('expandPath', () => {
  it('should expand ~ to home directory', () => {
    const result = expandPath('~/test/path');
    expect(result).toBe(path.join(os.homedir(), 'test/path'));
  });

  it('should expand ~/ at start only', () => {
    const result = expandPath('~/config/shoemaker');
    expect(result.startsWith(os.homedir())).toBe(true);
    expect(result.endsWith('config/shoemaker')).toBe(true);
  });

  it('should leave absolute paths unchanged', () => {
    const result = expandPath('/absolute/path');
    expect(result).toBe('/absolute/path');
  });

  it('should leave relative paths unchanged', () => {
    const result = expandPath('relative/path');
    expect(result).toBe('relative/path');
  });

  it('should handle ~ only', () => {
    const result = expandPath('~');
    expect(result).toBe(os.homedir());
  });
});

describe('getConfigPaths', () => {
  it('should return user and project config paths', () => {
    const paths = getConfigPaths();

    expect(paths.user).toContain('.config/shoemaker/config.toml');
    expect(paths.project).toContain('.shoemaker.toml');
  });

  it('should use home directory for user config', () => {
    const paths = getConfigPaths();
    expect(paths.user.startsWith(os.homedir())).toBe(true);
  });

  it('should use cwd for project config', () => {
    const paths = getConfigPaths();
    expect(paths.project.startsWith(process.cwd())).toBe(true);
  });
});
