/**
 * Extractor tests
 */

import { describe, it, expect } from 'vitest';
import { isRawFormat, isDecodedFormat } from '../../src/core/extractor.js';

describe('isRawFormat', () => {
  it('should identify Sony ARW', () => {
    expect(isRawFormat('image.arw')).toBe(true);
    expect(isRawFormat('IMAGE.ARW')).toBe(true);
    expect(isRawFormat('/path/to/image.ARW')).toBe(true);
  });

  it('should identify Canon CR2/CR3', () => {
    expect(isRawFormat('image.cr2')).toBe(true);
    expect(isRawFormat('image.CR3')).toBe(true);
  });

  it('should identify Nikon NEF', () => {
    expect(isRawFormat('image.nef')).toBe(true);
    expect(isRawFormat('image.NEF')).toBe(true);
  });

  it('should identify Fujifilm RAF', () => {
    expect(isRawFormat('image.raf')).toBe(true);
  });

  it('should identify DNG', () => {
    expect(isRawFormat('image.dng')).toBe(true);
    expect(isRawFormat('DJI_0001.DNG')).toBe(true);
  });

  it('should not identify decoded formats as RAW', () => {
    expect(isRawFormat('image.jpg')).toBe(false);
    expect(isRawFormat('image.jpeg')).toBe(false);
    expect(isRawFormat('image.png')).toBe(false);
    expect(isRawFormat('image.tiff')).toBe(false);
  });

  it('should handle files without extensions', () => {
    expect(isRawFormat('image')).toBe(false);
    expect(isRawFormat('/path/to/image')).toBe(false);
  });
});

describe('isDecodedFormat', () => {
  it('should identify JPEG', () => {
    expect(isDecodedFormat('image.jpg')).toBe(true);
    expect(isDecodedFormat('image.jpeg')).toBe(true);
    expect(isDecodedFormat('IMAGE.JPG')).toBe(true);
  });

  it('should identify PNG', () => {
    expect(isDecodedFormat('image.png')).toBe(true);
    expect(isDecodedFormat('image.PNG')).toBe(true);
  });

  it('should identify TIFF', () => {
    expect(isDecodedFormat('image.tif')).toBe(true);
    expect(isDecodedFormat('image.tiff')).toBe(true);
  });

  it('should identify WebP', () => {
    expect(isDecodedFormat('image.webp')).toBe(true);
  });

  it('should identify HEIC', () => {
    expect(isDecodedFormat('image.heic')).toBe(true);
    expect(isDecodedFormat('image.heif')).toBe(true);
  });

  it('should not identify RAW formats as decoded', () => {
    expect(isDecodedFormat('image.arw')).toBe(false);
    expect(isDecodedFormat('image.cr2')).toBe(false);
    expect(isDecodedFormat('image.nef')).toBe(false);
  });

  it('should handle files without extensions', () => {
    expect(isDecodedFormat('image')).toBe(false);
  });
});
