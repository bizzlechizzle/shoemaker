/**
 * info command
 *
 * Display embedded preview information for an image file,
 * or video metadata for a video file.
 */

import { Command } from 'commander';
import path from 'path';
import { analyzeEmbeddedPreviews, isVideoFormat } from '../../core/extractor.js';
import { probeVideo, checkFfprobeAvailable } from '../../core/ffprobe.js';
import { readThumbnailInfo } from '../../services/xmp-updater.js';

export const infoCommand = new Command('info')
  .description('Show embedded preview and thumbnail information')
  .argument('<file>', 'Image or video file to analyze')
  .option('--json', 'Output as JSON', false)
  .action(async (filePath: string, options) => {
    try {
      const thumbnailInfo = await readThumbnailInfo(filePath);

      // Handle video files differently
      if (isVideoFormat(filePath)) {
        await showVideoInfo(filePath, thumbnailInfo, options);
        return;
      }

      // Image file handling
      const analysis = await analyzeEmbeddedPreviews(filePath);

      if (options.json) {
        console.log(JSON.stringify({ analysis, thumbnailInfo }, null, 2));
        return;
      }

      console.log(`\n${path.basename(filePath)}`);
      console.log('═'.repeat(50));

      console.log('\nEmbedded Previews:');
      printPreviewInfo('  JpgFromRaw', analysis.jpgFromRaw);
      printPreviewInfo('  PreviewImage', analysis.previewImage);
      printPreviewInfo('  OtherImage', analysis.otherImage);
      printPreviewInfo('  ThumbnailImage', analysis.thumbnailImage);

      if (analysis.bestPreview) {
        console.log(`\n  Best: ${analysis.bestPreview.type} (${analysis.bestPreview.width}x${analysis.bestPreview.height})`);
      } else {
        console.log('\n  Best: None found');
      }

      console.log(`  Needs RAW decode: ${analysis.needsRawDecode ? 'Yes' : 'No'}`);

      if (thumbnailInfo.exists) {
        console.log('\nGenerated Thumbnails:');
        console.log(`  Generated at: ${thumbnailInfo.generatedAt}`);
        console.log(`  Method: ${thumbnailInfo.method}`);
        if (thumbnailInfo.thumbnails) {
          for (const thumb of thumbnailInfo.thumbnails) {
            console.log(`  ${thumb.size}: ${thumb.resolution}px ${thumb.format} (${formatBytes(thumb.bytes)})`);
          }
        }
      } else {
        console.log('\nGenerated Thumbnails: None');
      }

      console.log('');
    } catch (err) {
      console.error(`Error: ${(err as Error).message}`);
      process.exitCode = 1;
    }
  });

async function showVideoInfo(
  filePath: string,
  thumbnailInfo: Awaited<ReturnType<typeof readThumbnailInfo>>,
  options: { json?: boolean }
): Promise<void> {
  const ffprobeAvailable = await checkFfprobeAvailable();

  if (!ffprobeAvailable) {
    console.error('Error: FFprobe not found. Install FFmpeg to analyze video files.');
    process.exitCode = 1;
    return;
  }

  const videoInfo = await probeVideo(filePath);

  if (options.json) {
    console.log(JSON.stringify({ videoInfo, thumbnailInfo }, null, 2));
    return;
  }

  console.log(`\n${path.basename(filePath)}`);
  console.log('═'.repeat(50));

  console.log('\nVideo Information:');
  console.log(`  Duration: ${formatDuration(videoInfo.duration)}`);
  console.log(`  Resolution: ${videoInfo.width}x${videoInfo.height}`);
  console.log(`  Frame Rate: ${videoInfo.frameRate.toFixed(2)} fps`);
  console.log(`  Codec: ${videoInfo.codec}`);
  if (videoInfo.bitrate) {
    console.log(`  Bitrate: ${formatBitrate(videoInfo.bitrate)}`);
  }

  // Video characteristics
  const characteristics: string[] = [];
  if (videoInfo.isInterlaced) characteristics.push('Interlaced');
  if (videoInfo.isHdr) characteristics.push('HDR');
  if (videoInfo.rotation) characteristics.push(`Rotated ${videoInfo.rotation}°`);
  if (characteristics.length > 0) {
    console.log(`  Characteristics: ${characteristics.join(', ')}`);
  }

  // Audio info
  if (videoInfo.audio) {
    console.log('\nAudio:');
    console.log(`  Codec: ${videoInfo.audio.codec}`);
    console.log(`  Channels: ${videoInfo.audio.channels}`);
    console.log(`  Sample Rate: ${videoInfo.audio.sampleRate} Hz`);
  } else {
    console.log('\nAudio: None');
  }

  // Thumbnail info
  if (thumbnailInfo.exists) {
    console.log('\nGenerated Thumbnails:');
    console.log(`  Generated at: ${thumbnailInfo.generatedAt}`);
    console.log(`  Method: ${thumbnailInfo.method}`);
    if (thumbnailInfo.thumbnails) {
      for (const thumb of thumbnailInfo.thumbnails) {
        console.log(`  ${thumb.size}: ${thumb.resolution}px ${thumb.format} (${formatBytes(thumb.bytes)})`);
      }
    }
  } else {
    console.log('\nGenerated Thumbnails: None');
  }

  console.log('');
}

function printPreviewInfo(label: string, info: { exists: boolean; width?: number; height?: number; length?: number }): void {
  if (info.exists && info.width && info.height) {
    const size = info.length ? ` (${formatBytes(info.length)})` : '';
    console.log(`${label}: ${info.width}x${info.height}${size}`);
  } else {
    console.log(`${label}: Not present`);
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

function formatDuration(seconds: number): string {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 10);

  if (hrs > 0) {
    return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms}`;
  }
  return `${mins}:${secs.toString().padStart(2, '0')}.${ms} (${seconds.toFixed(1)}s)`;
}

function formatBitrate(bps: number): string {
  if (bps < 1000) return `${bps} bps`;
  if (bps < 1000000) return `${(bps / 1000).toFixed(1)} Kbps`;
  return `${(bps / 1000000).toFixed(1)} Mbps`;
}
