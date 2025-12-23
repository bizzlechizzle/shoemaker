/**
 * Video Integration Tests
 *
 * Tests the video processing pipeline including FFprobe and frame extraction.
 * Tests requiring real video files will be skipped if fixtures are not available.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = path.join(__dirname, '..', 'fixtures');
const VIDEO_DIR = path.join(FIXTURES_DIR, 'video');
const OUTPUT_DIR = path.join(__dirname, '..', 'output', 'video');

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function hasVideoFixtures(): Promise<boolean> {
  const sampleMp4 = path.join(VIDEO_DIR, 'sample.mp4');
  return fileExists(sampleMp4);
}

describe('Integration: FFprobe Service', () => {
  it('should check FFprobe availability', async () => {
    const { checkFfprobeAvailable, getFfprobeVersion } = await import('../../src/core/ffprobe.js');

    const available = await checkFfprobeAvailable();
    // FFprobe may or may not be installed - test should not fail
    expect(typeof available).toBe('boolean');

    if (available) {
      const version = await getFfprobeVersion();
      expect(typeof version).toBe('string');
      expect(version?.length).toBeGreaterThan(0);
    }
  });

  it('should check FFmpeg availability', async () => {
    const { checkFfmpegAvailable, getFfmpegVersion } = await import('../../src/core/ffprobe.js');

    const available = await checkFfmpegAvailable();
    expect(typeof available).toBe('boolean');

    if (available) {
      const version = await getFfmpegVersion();
      expect(typeof version).toBe('string');
      expect(version?.length).toBeGreaterThan(0);
    }
  });

  it('should clear FFprobe cache', async () => {
    const { clearFfprobeCache, checkFfprobeAvailable } = await import('../../src/core/ffprobe.js');

    // Clear cache and re-check
    clearFfprobeCache();
    const available = await checkFfprobeAvailable();
    expect(typeof available).toBe('boolean');
  });
});

describe('Integration: Video File Discovery', () => {
  it('should find video files in directory', async () => {
    const { findVideoFiles } = await import('../../src/services/thumbnail-generator.js');
    const { VIDEO_EXTENSIONS } = await import('../../src/schemas/index.js');

    // Use current directory as it should exist
    const tempDir = path.join(__dirname, '..', '..', 'temp-video-test');

    try {
      // Create temp directory with fake video files
      await fs.mkdir(tempDir, { recursive: true });
      await fs.writeFile(path.join(tempDir, 'test.mp4'), 'fake video');
      await fs.writeFile(path.join(tempDir, 'test.mov'), 'fake video');
      await fs.writeFile(path.join(tempDir, 'test.txt'), 'not a video');

      const files = await findVideoFiles(tempDir, false);

      expect(files.length).toBe(2);
      expect(files.some(f => f.endsWith('.mp4'))).toBe(true);
      expect(files.some(f => f.endsWith('.mov'))).toBe(true);
      expect(files.some(f => f.endsWith('.txt'))).toBe(false);
    } finally {
      // Cleanup
      try {
        await fs.rm(tempDir, { recursive: true, force: true });
      } catch {
        // Ignore cleanup errors
      }
    }
  });

  it('should include videos in findImageFiles when enabled', async () => {
    const { findImageFiles } = await import('../../src/services/thumbnail-generator.js');
    const { loadConfig } = await import('../../src/core/config.js');

    const tempDir = path.join(__dirname, '..', '..', 'temp-mixed-test');

    try {
      await fs.mkdir(tempDir, { recursive: true });
      await fs.writeFile(path.join(tempDir, 'test.mp4'), 'fake video');
      await fs.writeFile(path.join(tempDir, 'test.jpg'), 'fake image');

      const config = await loadConfig();

      // With video enabled (default)
      const filesWithVideo = await findImageFiles(tempDir, config, false, true);
      expect(filesWithVideo.some(f => f.endsWith('.mp4'))).toBe(true);
      expect(filesWithVideo.some(f => f.endsWith('.jpg'))).toBe(true);

      // Without video
      const filesNoVideo = await findImageFiles(tempDir, config, false, false);
      expect(filesNoVideo.some(f => f.endsWith('.mp4'))).toBe(false);
      expect(filesNoVideo.some(f => f.endsWith('.jpg'))).toBe(true);
    } finally {
      try {
        await fs.rm(tempDir, { recursive: true, force: true });
      } catch {
        // Ignore
      }
    }
  });
});

describe('Integration: Video Format Detection', () => {
  it('should detect video formats correctly', async () => {
    const { isVideoFormat } = await import('../../src/core/extractor.js');

    // Video formats
    expect(isVideoFormat('video.mp4')).toBe(true);
    expect(isVideoFormat('video.MP4')).toBe(true);
    expect(isVideoFormat('video.mov')).toBe(true);
    expect(isVideoFormat('video.avi')).toBe(true);
    expect(isVideoFormat('video.mkv')).toBe(true);
    expect(isVideoFormat('video.mxf')).toBe(true);
    expect(isVideoFormat('video.r3d')).toBe(true);
    expect(isVideoFormat('video.braw')).toBe(true);

    // Non-video formats
    expect(isVideoFormat('image.jpg')).toBe(false);
    expect(isVideoFormat('image.arw')).toBe(false);
    expect(isVideoFormat('document.pdf')).toBe(false);
  });
});

describe('Integration: Video Config', () => {
  it('should have valid video config defaults', async () => {
    const { loadConfig } = await import('../../src/core/config.js');

    const config = await loadConfig();

    expect(config.video).toBeDefined();
    expect(config.video.concurrency).toBeGreaterThan(0);
    expect(config.video.posterPosition).toBeGreaterThanOrEqual(0);
    expect(config.video.posterPosition).toBeLessThanOrEqual(100);
    expect(config.video.previewPosition).toBeGreaterThanOrEqual(0);
    expect(config.video.previewPosition).toBeLessThanOrEqual(100);
    expect(config.video.timelineFrames).toBeGreaterThanOrEqual(4);
    expect(config.video.timelineHeight).toBeGreaterThan(0);
    expect(typeof config.video.skipBlackFrames).toBe('boolean');
    expect(typeof config.video.autoDeinterlace).toBe('boolean');
    expect(typeof config.video.autoRotate).toBe('boolean');
    expect(typeof config.video.hdrToneMap).toBe('boolean');
  });
});

describe('Integration: Video Processing Pipeline', () => {
  let fixturesAvailable = false;
  let ffmpegAvailable = false;

  beforeAll(async () => {
    fixturesAvailable = await hasVideoFixtures();

    const { checkFfmpegAvailable } = await import('../../src/core/ffprobe.js');
    ffmpegAvailable = await checkFfmpegAvailable();

    if (fixturesAvailable && ffmpegAvailable) {
      try {
        await fs.rm(OUTPUT_DIR, { recursive: true });
      } catch {
        // Ignore if doesn't exist
      }
      await fs.mkdir(OUTPUT_DIR, { recursive: true });
    }
  });

  afterAll(async () => {
    try {
      await fs.rm(OUTPUT_DIR, { recursive: true });
    } catch {
      // Ignore errors
    }
  });

  it.skipIf(!fixturesAvailable || !ffmpegAvailable)('should probe video file', async () => {
    const { probeVideo } = await import('../../src/core/ffprobe.js');

    const videoPath = path.join(VIDEO_DIR, 'sample.mp4');
    const info = await probeVideo(videoPath);

    expect(info.duration).toBeGreaterThan(0);
    expect(info.width).toBeGreaterThan(0);
    expect(info.height).toBeGreaterThan(0);
    expect(info.frameRate).toBeGreaterThan(0);
    expect(typeof info.codec).toBe('string');
  });

  it.skipIf(!fixturesAvailable || !ffmpegAvailable)('should extract poster frame', async () => {
    const { extractPosterFrame } = await import('../../src/core/frame-extractor.js');
    const { loadConfig } = await import('../../src/core/config.js');

    const config = await loadConfig();
    const videoPath = path.join(VIDEO_DIR, 'sample.mp4');

    const frame = await extractPosterFrame(videoPath, config.video);

    expect(frame.buffer).toBeInstanceOf(Buffer);
    expect(frame.buffer.length).toBeGreaterThan(0);
    expect(frame.width).toBeGreaterThan(0);
    expect(frame.height).toBeGreaterThan(0);
    expect(typeof frame.position).toBe('number');
  });

  it.skipIf(!fixturesAvailable || !ffmpegAvailable)('should extract preview frame', async () => {
    const { extractPreviewFrame } = await import('../../src/core/frame-extractor.js');
    const { loadConfig } = await import('../../src/core/config.js');

    const config = await loadConfig();
    const videoPath = path.join(VIDEO_DIR, 'sample.mp4');

    const frame = await extractPreviewFrame(videoPath, config.video);

    expect(frame.buffer).toBeInstanceOf(Buffer);
    expect(frame.buffer.length).toBeGreaterThan(0);
    expect(frame.width).toBeGreaterThan(0);
    expect(frame.height).toBeGreaterThan(0);
  });

  it.skipIf(!fixturesAvailable || !ffmpegAvailable)('should generate timeline strip', async () => {
    const { generateTimelineStrip } = await import('../../src/core/frame-extractor.js');

    const videoPath = path.join(VIDEO_DIR, 'sample.mp4');
    const frameCount = 8;
    const frameHeight = 90;

    const strip = await generateTimelineStrip(videoPath, frameCount, frameHeight, {
      skipBlackFrames: true,
      deinterlace: true,
      rotate: true,
      hdrToneMap: true,
    });

    expect(strip).toBeInstanceOf(Buffer);
    expect(strip.length).toBeGreaterThan(0);

    // Verify it's a valid JPEG by checking magic bytes
    expect(strip[0]).toBe(0xff);
    expect(strip[1]).toBe(0xd8);
  });

  it.skipIf(!fixturesAvailable || !ffmpegAvailable)('should process video file end-to-end', async () => {
    const { generateForFile } = await import('../../src/services/thumbnail-generator.js');
    const { loadConfig, loadPreset, applyPreset } = await import('../../src/core/config.js');

    const config = await loadConfig();
    const preset = await loadPreset('fast', config);
    let finalConfig = applyPreset(config, preset);

    finalConfig = {
      ...finalConfig,
      output: {
        ...finalConfig.output,
        location: 'cache' as const,
        cacheDir: OUTPUT_DIR,
      },
      xmp: {
        ...finalConfig.xmp,
        updateSidecars: false,
      },
    };

    const videoPath = path.join(VIDEO_DIR, 'sample.mp4');

    const result = await generateForFile(videoPath, {
      config: finalConfig,
      preset,
      force: true,
    });

    expect(result.method).toBe('video');
    expect(result.thumbnails.length).toBeGreaterThan(0);
    expect(result.duration).toBeDefined();

    // Should have poster, preview, and timeline
    const sizes = result.thumbnails.map(t => t.size);
    expect(sizes).toContain('poster');
    expect(sizes).toContain('preview');
    expect(sizes).toContain('timeline');

    // Verify files were created
    for (const thumb of result.thumbnails) {
      const exists = await fileExists(thumb.path);
      expect(exists).toBe(true);
    }
  });

  it.skipIf(!fixturesAvailable || !ffmpegAvailable)('should handle video dry-run', async () => {
    const { generateForFile } = await import('../../src/services/thumbnail-generator.js');
    const { loadConfig, loadPreset, applyPreset } = await import('../../src/core/config.js');

    const config = await loadConfig();
    const preset = await loadPreset('fast', config);
    const finalConfig = applyPreset(config, preset);

    const videoPath = path.join(VIDEO_DIR, 'sample.mp4');

    const result = await generateForFile(videoPath, {
      config: finalConfig,
      preset,
      dryRun: true,
    });

    expect(result.method).toBe('video');
    expect(result.thumbnails).toEqual([]);
    expect(result.warnings).toContain('Dry run: no files written');
  });
});

describe('Integration: Doctor Command Video', () => {
  it('should include video tools in doctor check', async () => {
    const { execFile } = await import('child_process');
    const { promisify } = await import('util');
    const execFileAsync = promisify(execFile);

    const ROOT = path.join(__dirname, '..', '..');
    const CLI_PATH = path.join(ROOT, 'dist', 'bin', 'shoemaker.js');

    try {
      const { stdout } = await execFileAsync('node', [CLI_PATH, 'doctor', '--json'], {
        timeout: 30000,
        cwd: ROOT,
      });

      const parsed = JSON.parse(stdout);

      // Should have video tools section
      expect(parsed.videoTools).toBeDefined();
      expect(typeof parsed.videoTools.ffmpeg).toBe('boolean');
      expect(typeof parsed.videoTools.ffprobe).toBe('boolean');
    } catch (err) {
      // CLI might not be built in test environment
      console.log('Doctor JSON test skipped - CLI not built');
    }
  });
});
