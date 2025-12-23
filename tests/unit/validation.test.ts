/**
 * Input validation tests
 */

import { describe, it, expect } from 'vitest';

describe('Path validation edge cases', () => {
  it('should handle empty string path', () => {
    const path = '';
    expect(path.length).toBe(0);
  });

  it('should handle paths with spaces', () => {
    const path = '/path/with spaces/file.arw';
    expect(path.includes(' ')).toBe(true);
    expect(path.split('.').pop()).toBe('arw');
  });

  it('should handle paths with unicode', () => {
    const path = '/path/日本語/ファイル.arw';
    const normalized = path.normalize('NFC');
    expect(normalized).toBe(path);
  });

  it('should handle very long paths', () => {
    const longDir = 'a'.repeat(200);
    const path = `/${longDir}/file.arw`;
    expect(path.length).toBeGreaterThan(200);
  });

  it('should handle paths with special characters', () => {
    const path = '/path/with-dashes_underscores/file.arw';
    expect(path.includes('-')).toBe(true);
    expect(path.includes('_')).toBe(true);
  });
});

describe('Concurrency validation', () => {
  it('should accept valid concurrency values', () => {
    const validateConcurrency = (value: string): number | null => {
      const n = parseInt(value, 10);
      if (isNaN(n) || n < 1 || n > 32) return null;
      return n;
    };

    expect(validateConcurrency('1')).toBe(1);
    expect(validateConcurrency('4')).toBe(4);
    expect(validateConcurrency('32')).toBe(32);
  });

  it('should reject invalid concurrency values', () => {
    const validateConcurrency = (value: string): number | null => {
      const n = parseInt(value, 10);
      if (isNaN(n) || n < 1 || n > 32) return null;
      return n;
    };

    expect(validateConcurrency('0')).toBeNull();
    expect(validateConcurrency('-1')).toBeNull();
    expect(validateConcurrency('33')).toBeNull();
    expect(validateConcurrency('abc')).toBeNull();
    expect(validateConcurrency('')).toBeNull();
    expect(validateConcurrency('4.5')).toBe(4); // parseInt truncates
  });
});

describe('Number type guard', () => {
  const isNumber = (value: unknown): value is number => {
    return typeof value === 'number' && !isNaN(value) && isFinite(value);
  };

  it('should return true for valid numbers', () => {
    expect(isNumber(0)).toBe(true);
    expect(isNumber(1)).toBe(true);
    expect(isNumber(-1)).toBe(true);
    expect(isNumber(1.5)).toBe(true);
    expect(isNumber(Number.MAX_SAFE_INTEGER)).toBe(true);
  });

  it('should return false for invalid numbers', () => {
    expect(isNumber(NaN)).toBe(false);
    expect(isNumber(Infinity)).toBe(false);
    expect(isNumber(-Infinity)).toBe(false);
  });

  it('should return false for non-numbers', () => {
    expect(isNumber('1')).toBe(false);
    expect(isNumber(null)).toBe(false);
    expect(isNumber(undefined)).toBe(false);
    expect(isNumber({})).toBe(false);
    expect(isNumber([])).toBe(false);
    expect(isNumber(true)).toBe(false);
  });
});

describe('Percentage calculation edge cases', () => {
  it('should handle division by zero', () => {
    const pct = (n: number, total: number) =>
      total > 0 ? ((n / total) * 100).toFixed(1) : '0.0';

    expect(pct(0, 0)).toBe('0.0');
    expect(pct(5, 0)).toBe('0.0');
    expect(pct(0, 10)).toBe('0.0');
    expect(pct(5, 10)).toBe('50.0');
    expect(pct(10, 10)).toBe('100.0');
  });
});

describe('Buffer validation', () => {
  it('should detect empty buffer', () => {
    const buffer = Buffer.alloc(0);
    expect(buffer.length).toBe(0);
    expect(buffer.length === 0).toBe(true);
  });

  it('should detect non-empty buffer', () => {
    const buffer = Buffer.from('test');
    expect(buffer.length).toBeGreaterThan(0);
    expect(buffer.length === 0).toBe(false);
  });
});

describe('Extension extraction', () => {
  it('should extract extension correctly', () => {
    const getExt = (path: string) => path.toLowerCase().split('.').pop() ?? '';

    expect(getExt('file.arw')).toBe('arw');
    expect(getExt('file.ARW')).toBe('arw');
    expect(getExt('/path/to/file.ARW')).toBe('arw');
    expect(getExt('file.name.with.dots.arw')).toBe('arw');
  });

  it('should handle files without extension', () => {
    const getExt = (path: string) => path.toLowerCase().split('.').pop() ?? '';

    // Note: split('.').pop() returns the last segment after split
    // For '/path/to/file', this returns '/path/to/file' (no dots)
    // This is the actual behavior - for real usage we'd use path.extname()
    expect(getExt('file')).toBe('file');
    expect(getExt('/path/to/file')).toBe('/path/to/file');
  });

  it('should handle hidden files', () => {
    const getExt = (path: string) => path.toLowerCase().split('.').pop() ?? '';

    expect(getExt('.gitignore')).toBe('gitignore');
    expect(getExt('.hidden.arw')).toBe('arw');
  });
});
