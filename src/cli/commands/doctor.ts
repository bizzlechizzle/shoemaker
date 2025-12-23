/**
 * doctor command
 *
 * Check system dependencies and capabilities.
 */

import { Command } from 'commander';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';
import { getSharpCapabilities } from '../../core/resizer.js';
import { checkFfprobeAvailable, getFfprobeVersion, checkFfmpegAvailable, getFfmpegVersion } from '../../core/ffprobe.js';

const execFileAsync = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));

interface ToolInfo {
  name: string;
  available: boolean;
  version?: string;
  path?: string;
}

export const doctorCommand = new Command('doctor')
  .description('Check system dependencies and capabilities')
  .option('--json', 'Output as JSON', false)
  .action(async (options) => {
    const pkg = JSON.parse(readFileSync(path.join(__dirname, '../../../package.json'), 'utf-8'));

    const decoders = await checkDecoders();
    const metadata = await checkMetadataTools();
    const videoTools = await checkVideoTools();
    const sharp = getSharpCapabilities();

    if (options.json) {
      console.log(JSON.stringify({
        version: pkg.version,
        decoders,
        metadata,
        videoTools,
        sharp,
      }, null, 2));
      return;
    }

    console.log(`\nShoemaker v${pkg.version} — System Check\n`);

    console.log('RAW Decoders:');
    printToolStatus('  embedded', { name: 'embedded', available: true, version: 'Built-in' });
    printToolStatus('  libraw', { name: 'libraw', available: true, version: 'WASM (bundled)' });
    for (const decoder of decoders) {
      printToolStatus(`  ${decoder.name}`, decoder);
    }

    console.log('\nVideo Tools:');
    for (const tool of videoTools) {
      printToolStatus(`  ${tool.name}`, tool);
    }

    console.log('\nMetadata Tools:');
    printToolStatus('  exiftool', { name: 'exiftool', available: true, version: 'Bundled via exiftool-vendored' });
    for (const tool of metadata) {
      printToolStatus(`  ${tool.name}`, tool);
    }

    console.log('\nImage Processing:');
    console.log(`  ✓ sharp             v${sharp.version} (libvips)`);
    console.log(`  ✓ Formats: ${sharp.formats.output.join(', ')}`);

    // Check format support
    console.log('\nFormat Support:');
    console.log('  ✓ WebP              Encode/Decode');
    console.log('  ✓ JPEG              Encode/Decode');
    console.log('  ✓ PNG               Encode/Decode');
    if (sharp.formats.output.includes('avif')) {
      console.log('  ✓ AVIF              Encode/Decode');
    } else {
      console.log('  ✗ AVIF              Not available');
    }

    // Video support status
    const hasVideo = videoTools.every(t => t.available);
    console.log('\nVideo Support:');
    if (hasVideo) {
      console.log('  ✓ Video thumbnailing enabled (FFmpeg + FFprobe available)');
    } else {
      console.log('  ✗ Video thumbnailing disabled (install FFmpeg to enable)');
    }

    // Recommendations
    console.log('\nRecommended Presets:');
    const hasRt = decoders.find(d => d.name === 'rawtherapee-cli')?.available;
    const hasDt = decoders.find(d => d.name === 'darktable-cli')?.available;

    console.log('  • fast-import    → embedded + libraw fallback');
    if (hasRt) {
      console.log('  • high-quality   → rawtherapee-cli (best available)');
    } else if (hasDt) {
      console.log('  • high-quality   → darktable-cli (best available)');
    } else {
      console.log('  • high-quality   → libraw (install rawtherapee for better quality)');
    }
    console.log('  • portable       → libraw only (CI/CD safe)');

    console.log('\n✓ All core systems operational.\n');
  });

// Security: whitelist of allowed decoder commands
const ALLOWED_DECODER_COMMANDS = new Set(['rawtherapee-cli', 'darktable-cli', 'dcraw']);
const ALLOWED_METADATA_COMMANDS = new Set(['exiv2']);

async function checkDecoders(): Promise<ToolInfo[]> {
  const tools = [
    { name: 'rawtherapee-cli', command: 'rawtherapee-cli', versionArg: '--version' },
    { name: 'darktable-cli', command: 'darktable-cli', versionArg: '--version' },
    { name: 'dcraw', command: 'dcraw', versionArg: '' },
  ].filter(t => ALLOWED_DECODER_COMMANDS.has(t.command));

  const results: ToolInfo[] = [];

  for (const tool of tools) {
    try {
      const { stdout } = await execFileAsync('which', [tool.command]);
      const toolPath = stdout.trim();

      let version = 'Unknown';
      if (tool.versionArg) {
        try {
          const { stdout: versionOut, stderr: versionErr } = await execFileAsync(tool.command, [tool.versionArg]);
          const versionMatch = (versionOut || versionErr).match(/(\d+\.\d+(?:\.\d+)?)/);
          if (versionMatch && versionMatch[1]) {
            version = versionMatch[1];
          }
        } catch {
          // Version check failed, but tool exists
        }
      }

      results.push({
        name: tool.name,
        available: true,
        version,
        path: toolPath,
      });
    } catch {
      results.push({
        name: tool.name,
        available: false,
      });
    }
  }

  return results;
}

async function checkMetadataTools(): Promise<ToolInfo[]> {
  const tools = [
    { name: 'exiv2', command: 'exiv2', versionArg: '--version' },
  ].filter(t => ALLOWED_METADATA_COMMANDS.has(t.command));

  const results: ToolInfo[] = [];

  for (const tool of tools) {
    try {
      const { stdout } = await execFileAsync('which', [tool.command]);
      const toolPath = stdout.trim();

      let version = 'Unknown';
      try {
        const { stdout: versionOut } = await execFileAsync(tool.command, [tool.versionArg]);
        const versionMatch = versionOut.match(/(\d+\.\d+(?:\.\d+)?)/);
        if (versionMatch && versionMatch[1]) {
          version = versionMatch[1];
        }
      } catch {
        // Version check failed
      }

      results.push({
        name: tool.name,
        available: true,
        version,
        path: toolPath,
      });
    } catch {
      results.push({
        name: tool.name,
        available: false,
      });
    }
  }

  return results;
}

async function checkVideoTools(): Promise<ToolInfo[]> {
  const results: ToolInfo[] = [];

  // Check FFmpeg
  const ffmpegAvailable = await checkFfmpegAvailable();
  const ffmpegVersion = ffmpegAvailable ? await getFfmpegVersion() : null;
  results.push({
    name: 'ffmpeg',
    available: ffmpegAvailable,
    version: ffmpegVersion ?? undefined,
  });

  // Check FFprobe
  const ffprobeAvailable = await checkFfprobeAvailable();
  const ffprobeVersion = ffprobeAvailable ? await getFfprobeVersion() : null;
  results.push({
    name: 'ffprobe',
    available: ffprobeAvailable,
    version: ffprobeVersion ?? undefined,
  });

  return results;
}

function printToolStatus(label: string, tool: ToolInfo): void {
  if (tool.available) {
    const version = tool.version ? `v${tool.version}` : '';
    const toolPath = tool.path ? ` (${tool.path})` : '';
    console.log(`  ✓ ${label.padEnd(18)} ${version}${toolPath}`);
  } else {
    console.log(`  ✗ ${label.padEnd(18)} Not found`);
  }
}
