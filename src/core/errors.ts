/**
 * Shoemaker Error Classes
 *
 * Structured error handling with error codes and recovery strategies.
 */

export enum ErrorCode {
  FILE_NOT_FOUND = 'FILE_NOT_FOUND',
  PERMISSION_DENIED = 'PERMISSION_DENIED',
  CORRUPT_FILE = 'CORRUPT_FILE',
  NO_PREVIEW = 'NO_PREVIEW',
  DECODE_FAILED = 'DECODE_FAILED',
  RESIZE_FAILED = 'RESIZE_FAILED',
  DISK_FULL = 'DISK_FULL',
  OUT_OF_MEMORY = 'OUT_OF_MEMORY',
  EXIFTOOL_ERROR = 'EXIFTOOL_ERROR',
  SHARP_ERROR = 'SHARP_ERROR',
  XMP_WRITE_FAILED = 'XMP_WRITE_FAILED',
  CONFIG_INVALID = 'CONFIG_INVALID',
  PRESET_NOT_FOUND = 'PRESET_NOT_FOUND',
  DECODER_NOT_AVAILABLE = 'DECODER_NOT_AVAILABLE',
  INVALID_PATH = 'INVALID_PATH',
  UNKNOWN = 'UNKNOWN',
}

export class ShoemakerError extends Error {
  constructor(
    message: string,
    public readonly code: ErrorCode,
    public readonly filePath?: string,
    public readonly recoverable: boolean = true,
    public readonly cause?: Error
  ) {
    super(message);
    this.name = 'ShoemakerError';
    Error.captureStackTrace?.(this, ShoemakerError);
  }

  toJSON() {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      filePath: this.filePath,
      recoverable: this.recoverable,
    };
  }
}

/**
 * Wrap unknown errors into ShoemakerError with appropriate code
 */
export function wrapError(err: unknown, filePath?: string): ShoemakerError {
  if (err instanceof ShoemakerError) {
    return err;
  }

  const message = err instanceof Error ? err.message : String(err);
  const cause = err instanceof Error ? err : undefined;

  // Detect specific error types from message
  if (message.includes('ENOENT') || message.includes('no such file')) {
    return new ShoemakerError(
      `File not found: ${filePath ?? 'unknown'}`,
      ErrorCode.FILE_NOT_FOUND,
      filePath,
      true,
      cause
    );
  }

  if (message.includes('EACCES') || message.includes('EPERM') || message.includes('permission denied')) {
    return new ShoemakerError(
      `Permission denied: ${filePath ?? 'unknown'}`,
      ErrorCode.PERMISSION_DENIED,
      filePath,
      true,
      cause
    );
  }

  if (message.includes('ENOSPC') || message.includes('no space left')) {
    return new ShoemakerError(
      'Disk full',
      ErrorCode.DISK_FULL,
      filePath,
      false, // Not recoverable
      cause
    );
  }

  if (message.includes('ENOMEM') || message.includes('heap') || message.includes('out of memory')) {
    return new ShoemakerError(
      'Out of memory',
      ErrorCode.OUT_OF_MEMORY,
      filePath,
      true, // Recoverable by reducing concurrency
      cause
    );
  }

  if (message.includes('Invalid image') || message.includes('corrupt') || message.includes('unsupported')) {
    return new ShoemakerError(
      `Corrupt or unsupported file: ${filePath ?? 'unknown'}`,
      ErrorCode.CORRUPT_FILE,
      filePath,
      true,
      cause
    );
  }

  if (message.includes('exiftool') || message.includes('ExifTool')) {
    return new ShoemakerError(
      `ExifTool error: ${message}`,
      ErrorCode.EXIFTOOL_ERROR,
      filePath,
      true,
      cause
    );
  }

  if (message.includes('sharp') || message.includes('vips')) {
    return new ShoemakerError(
      `Image processing error: ${message}`,
      ErrorCode.SHARP_ERROR,
      filePath,
      true,
      cause
    );
  }

  // Generic error
  return new ShoemakerError(
    message,
    ErrorCode.UNKNOWN,
    filePath,
    true,
    cause
  );
}

/**
 * Check if an error is recoverable (batch can continue)
 */
export function isRecoverable(err: unknown): boolean {
  if (err instanceof ShoemakerError) {
    return err.recoverable;
  }
  return true; // Assume unknown errors are recoverable
}

/**
 * Check if error requires stopping the batch
 */
export function shouldStopBatch(err: unknown): boolean {
  if (err instanceof ShoemakerError) {
    return err.code === ErrorCode.DISK_FULL;
  }
  return false;
}

/**
 * Check if error suggests reducing concurrency
 */
export function shouldReduceConcurrency(err: unknown): boolean {
  if (err instanceof ShoemakerError) {
    return err.code === ErrorCode.OUT_OF_MEMORY;
  }
  return false;
}
