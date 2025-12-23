/**
 * Shoemaker CLI
 *
 * Command-line interface for thumbnail generation.
 */

import { Command } from 'commander';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';
import { thumbCommand } from './commands/thumb.js';
import { infoCommand } from './commands/info.js';
import { cleanCommand } from './commands/clean.js';
import { statusCommand } from './commands/status.js';
import { doctorCommand } from './commands/doctor.js';
import { shutdownExiftool } from '../core/extractor.js';

// Get version from package.json
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(readFileSync(path.join(__dirname, '../../package.json'), 'utf-8'));

const program = new Command();

program
  .name('shoemaker')
  .description('A CLI that makes thumbnails from images, RAW files, and videos')
  .version(pkg.version);

// Register commands
program.addCommand(thumbCommand);
program.addCommand(infoCommand);
program.addCommand(cleanCommand);
program.addCommand(statusCommand);
program.addCommand(doctorCommand);

// Handle shutdown
process.on('SIGINT', async () => {
  await shutdownExiftool();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await shutdownExiftool();
  process.exit(0);
});

// Parse and execute
export async function run(argv: string[] = process.argv): Promise<void> {
  try {
    await program.parseAsync(argv);
  } finally {
    await shutdownExiftool();
  }
}

export { program };
