const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const toIco = require('to-ico');

const ICONS_DIR = './src-tauri/icons';

async function main() {
  console.log('Starting full icon creation...');
  
  const svg = fs.readFileSync('./mimic.svg');
  console.log('SVG loaded, size:', svg.length);
  
  // Generate all required icon sizes for Tauri
  const sizes = [
    { name: '32x32.png', size: 32 },
    { name: '64x64.png', size: 64 },
    { name: '128x128.png', size: 128 },
    { name: '128x128@2x.png', size: 256 }, // Retina display
    { name: 'icon.png', size: 256 }, // Default icon
  ];
  
  // Windows Store/Tile icons
  const storeSizes = [
    { name: 'Square30x30Logo.png', size: 30 },
    { name: 'Square44x44Logo.png', size: 44 },
    { name: 'Square71x71Logo.png', size: 71 },
    { name: 'Square89x89Logo.png', size: 89 },
    { name: 'Square107x107Logo.png', size: 107 },
    { name: 'Square142x142Logo.png', size: 142 },
    { name: 'Square150x150Logo.png', size: 150 },
    { name: 'Square284x284Logo.png', size: 284 },
    { name: 'Square310x310Logo.png', size: 310 },
    { name: 'StoreLogo.png', size: 50 },
  ];
  
  // Generate standard sizes
  console.log('\nGenerating standard icons...');
  for (const { name, size } of sizes) {
    const outputPath = path.join(ICONS_DIR, name);
    await sharp(svg).resize(size, size).png().toFile(outputPath);
    console.log(`  ✓ ${name} (${size}x${size})`);
  }
  
  // Generate Store/Tile icons
  console.log('\nGenerating Store/Tile icons...');
  for (const { name, size } of storeSizes) {
    const outputPath = path.join(ICONS_DIR, name);
    await sharp(svg).resize(size, size).png().toFile(outputPath);
    console.log(`  ✓ ${name} (${size}x${size})`);
  }
  
  // Create ICO file for Windows executable (multi-size)
  console.log('\nGenerating Windows ICO...');
  const buf256 = await sharp(svg).resize(256, 256).png().toBuffer();
  const buf128 = await sharp(svg).resize(128, 128).png().toBuffer();
  const buf64 = await sharp(svg).resize(64, 64).png().toBuffer();
  const buf48 = await sharp(svg).resize(48, 48).png().toBuffer();
  const buf32 = await sharp(svg).resize(32, 32).png().toBuffer();
  const buf16 = await sharp(svg).resize(16, 16).png().toBuffer();
  
  const ico = await toIco([buf256, buf128, buf64, buf48, buf32, buf16]);
  fs.writeFileSync(path.join(ICONS_DIR, 'icon.ico'), ico);
  console.log(`  ✓ icon.ico (multi-size: 256, 128, 64, 48, 32, 16)`);
  console.log(`  Size: ${(ico.length / 1024).toFixed(1)} KB`);
  
  console.log('\n✅ All icons generated successfully!');
  console.log('\nNext steps:');
  console.log('  1. Rebuild the Tauri app:');
  console.log('     .\\scripts\\build-installer.ps1');
  console.log('  2. The new icon will be embedded in Mimic AI.exe');
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(1);
});
