/**
 * info command
 *
 * Display embedded preview information for an image file.
 */

import { Command } from 'commander';
import path from 'path';
import { analyzeEmbeddedPreviews } from '../../core/extractor.js';
import { readThumbnailInfo } from '../../services/xmp-updater.js';

export const infoCommand = new Command('info')
  .description('Show embedded preview and thumbnail information')
  .argument('<file>', 'Image file to analyze')
  .option('--json', 'Output as JSON', false)
  .action(async (filePath: string, options) => {
    try {
      const analysis = await analyzeEmbeddedPreviews(filePath);
      const thumbnailInfo = await readThumbnailInfo(filePath);

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
