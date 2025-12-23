/**
 * FFprobe Module Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';

describe('Video Format Detection', () => {
  let isVideoFormat: (path: string) => boolean;

  beforeEach(async () => {
    const module = await import('../../src/core/extractor.js');
    isVideoFormat = module.isVideoFormat;
  });

  describe('isVideoFormat', () => {
    it('should identify common video formats', () => {
      expect(isVideoFormat('video.mp4')).toBe(true);
      expect(isVideoFormat('video.mov')).toBe(true);
      expect(isVideoFormat('video.avi')).toBe(true);
      expect(isVideoFormat('video.mkv')).toBe(true);
      expect(isVideoFormat('video.webm')).toBe(true);
    });

    it('should handle case insensitive extensions', () => {
      expect(isVideoFormat('video.MP4')).toBe(true);
      expect(isVideoFormat('video.MOV')).toBe(true);
      expect(isVideoFormat('video.Mkv')).toBe(true);
    });

    it('should identify professional video formats', () => {
      expect(isVideoFormat('video.mxf')).toBe(true);
      expect(isVideoFormat('video.mts')).toBe(true);
      expect(isVideoFormat('video.m2ts')).toBe(true);
    });

    it('should identify camera-specific formats', () => {
      expect(isVideoFormat('video.tod')).toBe(true);
      expect(isVideoFormat('video.mod')).toBe(true);
      expect(isVideoFormat('video.3gp')).toBe(true);
      expect(isVideoFormat('video.r3d')).toBe(true);
      expect(isVideoFormat('video.braw')).toBe(true);
    });

    it('should return false for image formats', () => {
      expect(isVideoFormat('image.jpg')).toBe(false);
      expect(isVideoFormat('image.png')).toBe(false);
      expect(isVideoFormat('image.arw')).toBe(false);
      expect(isVideoFormat('image.cr2')).toBe(false);
      expect(isVideoFormat('image.nef')).toBe(false);
    });

    it('should return false for non-media files', () => {
      expect(isVideoFormat('document.pdf')).toBe(false);
      expect(isVideoFormat('data.json')).toBe(false);
      expect(isVideoFormat('readme.md')).toBe(false);
      expect(isVideoFormat('script.ts')).toBe(false);
    });

    it('should handle files with multiple dots in name', () => {
      expect(isVideoFormat('my.video.file.mp4')).toBe(true);
      expect(isVideoFormat('holiday.2024.mov')).toBe(true);
    });

    it('should handle files without extension', () => {
      expect(isVideoFormat('noextension')).toBe(false);
    });
  });
});

describe('Video Schemas', () => {
  let VideoInfoSchema: import('zod').ZodObject<any>;
  let VideoConfigSchema: import('zod').ZodObject<any>;

  beforeEach(async () => {
    const module = await import('../../src/schemas/index.js');
    VideoInfoSchema = module.VideoInfoSchema;
    VideoConfigSchema = module.VideoConfigSchema;
  });

  describe('VideoInfoSchema', () => {
    it('should validate valid video info', () => {
      const validInfo = {
        duration: 120.5,
        width: 1920,
        height: 1080,
        frameRate: 29.97,
        codec: 'h264',
      };

      const result = VideoInfoSchema.safeParse(validInfo);
      expect(result.success).toBe(true);
    });

    it('should accept optional fields', () => {
      const infoWithOptional = {
        duration: 60,
        width: 1920,
        height: 1080,
        frameRate: 30,
        codec: 'h264',
        bitrate: 12000000,
        rotation: 90,
        isInterlaced: true,
        isHdr: false,
        audio: {
          codec: 'aac',
          channels: 2,
          sampleRate: 48000,
        },
      };

      const result = VideoInfoSchema.safeParse(infoWithOptional);
      expect(result.success).toBe(true);
    });

    it('should reject missing required fields', () => {
      const invalidInfo = {
        duration: 120,
        width: 1920,
        // missing height, frameRate, codec
      };

      const result = VideoInfoSchema.safeParse(invalidInfo);
      expect(result.success).toBe(false);
    });

    it('should validate audio sub-object', () => {
      const infoWithAudio = {
        duration: 60,
        width: 1920,
        height: 1080,
        frameRate: 30,
        codec: 'h264',
        audio: {
          codec: 'aac',
          channels: 2,
          sampleRate: 48000,
        },
      };

      const result = VideoInfoSchema.safeParse(infoWithAudio);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.audio.codec).toBe('aac');
        expect(result.data.audio.channels).toBe(2);
        expect(result.data.audio.sampleRate).toBe(48000);
      }
    });
  });

  describe('VideoConfigSchema', () => {
    it('should provide sensible defaults', () => {
      const result = VideoConfigSchema.parse({});

      expect(result.concurrency).toBe(2);
      expect(result.posterPosition).toBe(25);
      expect(result.previewPosition).toBe(50);
      expect(result.timelineFrames).toBe(8);
      expect(result.timelineHeight).toBe(90);
      expect(result.skipBlackFrames).toBe(true);
      expect(result.autoDeinterlace).toBe(true);
      expect(result.autoRotate).toBe(true);
      expect(result.hdrToneMap).toBe(true);
    });

    it('should validate custom values', () => {
      const customConfig = {
        concurrency: 4,
        posterPosition: 33,
        previewPosition: 66,
        timelineFrames: 12,
        timelineHeight: 120,
        skipBlackFrames: false,
        autoDeinterlace: false,
        autoRotate: true,
        hdrToneMap: false,
      };

      const result = VideoConfigSchema.safeParse(customConfig);
      expect(result.success).toBe(true);
      if (result.success) {
        // Check core values match (proxy gets default values)
        expect(result.data.concurrency).toEqual(customConfig.concurrency);
        expect(result.data.posterPosition).toEqual(customConfig.posterPosition);
        expect(result.data.previewPosition).toEqual(customConfig.previewPosition);
        expect(result.data.timelineFrames).toEqual(customConfig.timelineFrames);
        expect(result.data.proxy).toBeDefined();
        expect(result.data.proxy.enabled).toBe(false);
      }
    });

    it('should reject invalid concurrency values', () => {
      expect(VideoConfigSchema.safeParse({ concurrency: 0 }).success).toBe(false);
      expect(VideoConfigSchema.safeParse({ concurrency: 100 }).success).toBe(false);
      expect(VideoConfigSchema.safeParse({ concurrency: -1 }).success).toBe(false);
    });

    it('should reject invalid position values', () => {
      expect(VideoConfigSchema.safeParse({ posterPosition: -10 }).success).toBe(false);
      expect(VideoConfigSchema.safeParse({ posterPosition: 150 }).success).toBe(false);
      expect(VideoConfigSchema.safeParse({ previewPosition: 200 }).success).toBe(false);
    });

    it('should reject invalid timeline values', () => {
      expect(VideoConfigSchema.safeParse({ timelineFrames: 2 }).success).toBe(false);
      expect(VideoConfigSchema.safeParse({ timelineFrames: 100 }).success).toBe(false);
      expect(VideoConfigSchema.safeParse({ timelineHeight: 20 }).success).toBe(false);
      expect(VideoConfigSchema.safeParse({ timelineHeight: 500 }).success).toBe(false);
    });

    it('should accept boundary values', () => {
      const boundaryConfig = {
        concurrency: 1,
        posterPosition: 0,
        previewPosition: 100,
        timelineFrames: 4,
        timelineHeight: 40,
      };

      const result = VideoConfigSchema.safeParse(boundaryConfig);
      expect(result.success).toBe(true);
    });
  });
});

describe('VIDEO_EXTENSIONS constant', () => {
  let VIDEO_EXTENSIONS: readonly string[];

  beforeEach(async () => {
    const module = await import('../../src/schemas/index.js');
    VIDEO_EXTENSIONS = module.VIDEO_EXTENSIONS;
  });

  it('should include common video extensions', () => {
    expect(VIDEO_EXTENSIONS).toContain('mp4');
    expect(VIDEO_EXTENSIONS).toContain('mov');
    expect(VIDEO_EXTENSIONS).toContain('avi');
    expect(VIDEO_EXTENSIONS).toContain('mkv');
    expect(VIDEO_EXTENSIONS).toContain('webm');
  });

  it('should include professional formats', () => {
    expect(VIDEO_EXTENSIONS).toContain('mxf');
    expect(VIDEO_EXTENSIONS).toContain('mts');
    expect(VIDEO_EXTENSIONS).toContain('m2ts');
    expect(VIDEO_EXTENSIONS).toContain('mpg');
    expect(VIDEO_EXTENSIONS).toContain('mpeg');
  });

  it('should include camera-specific formats', () => {
    expect(VIDEO_EXTENSIONS).toContain('tod');
    expect(VIDEO_EXTENSIONS).toContain('mod');
    expect(VIDEO_EXTENSIONS).toContain('3gp');
    expect(VIDEO_EXTENSIONS).toContain('r3d');
    expect(VIDEO_EXTENSIONS).toContain('braw');
  });

  it('should not include image formats', () => {
    expect(VIDEO_EXTENSIONS).not.toContain('jpg');
    expect(VIDEO_EXTENSIONS).not.toContain('jpeg');
    expect(VIDEO_EXTENSIONS).not.toContain('png');
    expect(VIDEO_EXTENSIONS).not.toContain('arw');
    expect(VIDEO_EXTENSIONS).not.toContain('cr2');
  });

  it('should have expected number of formats', () => {
    // Check it's a reasonable number
    expect(VIDEO_EXTENSIONS.length).toBeGreaterThanOrEqual(15);
    expect(VIDEO_EXTENSIONS.length).toBeLessThanOrEqual(30);
  });
});

describe('Config with Video settings', () => {
  let ConfigSchema: import('zod').ZodObject<any>;

  beforeEach(async () => {
    const module = await import('../../src/schemas/index.js');
    ConfigSchema = module.ConfigSchema;
  });

  it('should include video config in full config', () => {
    const result = ConfigSchema.parse({});
    expect(result.video).toBeDefined();
    expect(result.video.concurrency).toBe(2);
    expect(result.video.posterPosition).toBe(25);
  });

  it('should allow overriding video config', () => {
    const result = ConfigSchema.parse({
      video: {
        concurrency: 4,
        timelineFrames: 12,
      },
    });

    expect(result.video.concurrency).toBe(4);
    expect(result.video.timelineFrames).toBe(12);
    // Defaults should still apply
    expect(result.video.posterPosition).toBe(25);
  });
});

describe('GenerationResult with video method', () => {
  let GenerationResultSchema: import('zod').ZodObject<any>;

  beforeEach(async () => {
    const module = await import('../../src/schemas/index.js');
    GenerationResultSchema = module.GenerationResultSchema;
  });

  it('should accept video as a valid method', () => {
    const videoResult = {
      source: '/path/to/video.mp4',
      method: 'video',
      thumbnails: [],
      warnings: [],
      duration: 1500,
    };

    const result = GenerationResultSchema.safeParse(videoResult);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.method).toBe('video');
    }
  });

  it('should still accept other methods', () => {
    const methods = ['extracted', 'decoded', 'direct', 'video'];

    for (const method of methods) {
      const result = GenerationResultSchema.safeParse({
        source: '/path/to/file',
        method,
        thumbnails: [],
        duration: 100,
      });
      expect(result.success).toBe(true);
    }
  });
});
