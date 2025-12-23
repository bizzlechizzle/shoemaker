/**
 * Error handling tests
 */

import { describe, it, expect } from 'vitest';
import {
  ShoemakerError,
  ErrorCode,
  wrapError,
  isRecoverable,
  shouldStopBatch,
  shouldReduceConcurrency,
} from '../../src/core/errors.js';

describe('ShoemakerError', () => {
  it('should create error with all properties', () => {
    const error = new ShoemakerError(
      'Test error',
      ErrorCode.FILE_NOT_FOUND,
      '/path/to/file',
      true
    );

    expect(error.message).toBe('Test error');
    expect(error.code).toBe(ErrorCode.FILE_NOT_FOUND);
    expect(error.filePath).toBe('/path/to/file');
    expect(error.recoverable).toBe(true);
    expect(error.name).toBe('ShoemakerError');
  });

  it('should serialize to JSON correctly', () => {
    const error = new ShoemakerError(
      'Test error',
      ErrorCode.CORRUPT_FILE,
      '/path/to/file',
      false
    );

    const json = error.toJSON();

    expect(json).toEqual({
      name: 'ShoemakerError',
      code: 'CORRUPT_FILE',
      message: 'Test error',
      filePath: '/path/to/file',
      recoverable: false,
    });
  });
});

describe('wrapError', () => {
  it('should return ShoemakerError unchanged', () => {
    const original = new ShoemakerError('Original', ErrorCode.NO_PREVIEW, '/file');
    const wrapped = wrapError(original);

    expect(wrapped).toBe(original);
  });

  it('should wrap ENOENT as FILE_NOT_FOUND', () => {
    const error = new Error('ENOENT: no such file or directory');
    const wrapped = wrapError(error, '/missing/file');

    expect(wrapped.code).toBe(ErrorCode.FILE_NOT_FOUND);
    expect(wrapped.filePath).toBe('/missing/file');
    expect(wrapped.recoverable).toBe(true);
  });

  it('should wrap ENOSPC as DISK_FULL (non-recoverable)', () => {
    const error = new Error('ENOSPC: no space left on device');
    const wrapped = wrapError(error);

    expect(wrapped.code).toBe(ErrorCode.DISK_FULL);
    expect(wrapped.recoverable).toBe(false);
  });

  it('should wrap ENOMEM as OUT_OF_MEMORY (recoverable)', () => {
    const error = new Error('ENOMEM: out of memory');
    const wrapped = wrapError(error);

    expect(wrapped.code).toBe(ErrorCode.OUT_OF_MEMORY);
    expect(wrapped.recoverable).toBe(true);
  });

  it('should wrap corrupt file errors', () => {
    const error = new Error('Invalid image format');
    const wrapped = wrapError(error, '/bad/file.arw');

    expect(wrapped.code).toBe(ErrorCode.CORRUPT_FILE);
  });

  it('should wrap unknown errors as UNKNOWN', () => {
    const error = new Error('Something unexpected');
    const wrapped = wrapError(error);

    expect(wrapped.code).toBe(ErrorCode.UNKNOWN);
    expect(wrapped.recoverable).toBe(true);
  });

  it('should handle non-Error objects', () => {
    const wrapped = wrapError('string error');

    expect(wrapped.code).toBe(ErrorCode.UNKNOWN);
    expect(wrapped.message).toBe('string error');
  });
});

describe('isRecoverable', () => {
  it('should return recoverable flag for ShoemakerError', () => {
    const recoverable = new ShoemakerError('Test', ErrorCode.NO_PREVIEW, undefined, true);
    const nonRecoverable = new ShoemakerError('Test', ErrorCode.DISK_FULL, undefined, false);

    expect(isRecoverable(recoverable)).toBe(true);
    expect(isRecoverable(nonRecoverable)).toBe(false);
  });

  it('should return true for unknown errors', () => {
    expect(isRecoverable(new Error('Unknown'))).toBe(true);
    expect(isRecoverable('string error')).toBe(true);
  });
});

describe('shouldStopBatch', () => {
  it('should return true for DISK_FULL', () => {
    const error = new ShoemakerError('Full', ErrorCode.DISK_FULL, undefined, false);
    expect(shouldStopBatch(error)).toBe(true);
  });

  it('should return false for other errors', () => {
    const error = new ShoemakerError('Not found', ErrorCode.FILE_NOT_FOUND);
    expect(shouldStopBatch(error)).toBe(false);
  });

  it('should return false for unknown errors', () => {
    expect(shouldStopBatch(new Error('Unknown'))).toBe(false);
  });
});

describe('shouldReduceConcurrency', () => {
  it('should return true for OUT_OF_MEMORY', () => {
    const error = new ShoemakerError('OOM', ErrorCode.OUT_OF_MEMORY);
    expect(shouldReduceConcurrency(error)).toBe(true);
  });

  it('should return false for other errors', () => {
    const error = new ShoemakerError('Not found', ErrorCode.FILE_NOT_FOUND);
    expect(shouldReduceConcurrency(error)).toBe(false);
  });
});
