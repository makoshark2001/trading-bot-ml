// cleanup-legacy-storage.js - Run this script to remove old files after migration
const fs = require('fs');
const path = require('path');

class LegacyCleanup {
    constructor(baseDir = 'data/ml') {
        this.baseDir = baseDir;
        this.legacyDirs = ['models', 'weights', 'training', 'predictions', 'features'];
        this.consolidatedDir = path.join(baseDir, 'consolidated');
    }
    
    // Check if migration was successful
    checkMigrationSuccess() {
        if (!fs.existsSync(this.consolidatedDir)) {
            console.log('❌ No consolidated directory found. Migration may not have run.');
            return false;
        }
        
        const consolidatedFiles = fs.readdirSync(this.consolidatedDir)
            .filter(f => f.endsWith('_complete.json'));
            
        if (consolidatedFiles.length === 0) {
            console.log('❌ No consolidated files found. Migration may not have run.');
            return false;
        }
        
        console.log(`✅ Found ${consolidatedFiles.length} consolidated files`);
        return true;
    }
    
    // List what would be deleted (dry run)
    analyzeLegacyFiles() {
        const analysis = {
            totalSize: 0,
            directories: {},
            wouldDelete: []
        };
        
        for (const dirName of this.legacyDirs) {
            const dirPath = path.join(this.baseDir, dirName);
            
            if (fs.existsSync(dirPath)) {
                const files = this.getDirectorySize(dirPath);
                analysis.directories[dirName] = files;
                analysis.totalSize += files.totalSize;
                analysis.wouldDelete.push(dirPath);
            }
        }
        
        return analysis;
    }
    
    getDirectorySize(dirPath) {
        let totalSize = 0;
        let fileCount = 0;
        
        const traverse = (currentPath) => {
            const items = fs.readdirSync(currentPath);
            
            items.forEach(item => {
                const itemPath = path.join(currentPath, item);
                const stats = fs.statSync(itemPath);
                
                if (stats.isDirectory()) {
                    traverse(itemPath);
                } else {
                    totalSize += stats.size;
                    fileCount++;
                }
            });
        };
        
        traverse(dirPath);
        
        return {
            totalSize,
            fileCount,
            sizeInMB: (totalSize / (1024 * 1024)).toFixed(2)
        };
    }
    
    // Actually delete the legacy files
    cleanupLegacyFiles(confirmed = false) {
        if (!confirmed) {
            console.log('❌ Cleanup not confirmed. Use cleanup(true) to actually delete files.');
            return false;
        }
        
        if (!this.checkMigrationSuccess()) {
            console.log('❌ Cannot cleanup - migration verification failed');
            return false;
        }
        
        let deletedDirs = 0;
        const backupDir = path.join(this.baseDir, `legacy_backup_${Date.now()}`);
        
        console.log(`📦 Creating final backup in: ${backupDir}`);
        fs.mkdirSync(backupDir, { recursive: true });
        
        for (const dirName of this.legacyDirs) {
            const dirPath = path.join(this.baseDir, dirName);
            
            if (fs.existsSync(dirPath)) {
                try {
                    // Create backup first
                    const backupPath = path.join(backupDir, dirName);
                    this.copyDirectory(dirPath, backupPath);
                    
                    // Delete original
                    this.removeDirectory(dirPath);
                    deletedDirs++;
                    
                    console.log(`🗑️  Deleted legacy directory: ${dirName}`);
                } catch (error) {
                    console.error(`❌ Failed to delete ${dirName}:`, error.message);
                }
            }
        }
        
        console.log(`✅ Cleanup completed. Deleted ${deletedDirs} directories.`);
        console.log(`💾 Backup available at: ${backupDir}`);
        
        return true;
    }
    
    copyDirectory(source, destination) {
        if (!fs.existsSync(destination)) {
            fs.mkdirSync(destination, { recursive: true });
        }
        
        const items = fs.readdirSync(source);
        
        items.forEach(item => {
            const sourcePath = path.join(source, item);
            const destPath = path.join(destination, item);
            
            if (fs.statSync(sourcePath).isDirectory()) {
                this.copyDirectory(sourcePath, destPath);
            } else {
                fs.copyFileSync(sourcePath, destPath);
            }
        });
    }
    
    removeDirectory(dirPath) {
        if (fs.existsSync(dirPath)) {
            fs.rmSync(dirPath, { recursive: true, force: true });
        }
    }
    
    // Interactive cleanup
    async interactiveCleanup() {
        console.log('🔍 Analyzing legacy storage...\n');
        
        if (!this.checkMigrationSuccess()) {
            return;
        }
        
        const analysis = this.analyzeLegacyFiles();
        
        if (analysis.wouldDelete.length === 0) {
            console.log('✅ No legacy files found to clean up.');
            return;
        }
        
        console.log('📊 Legacy Storage Analysis:');
        console.log(`   Total directories: ${analysis.wouldDelete.length}`);
        console.log(`   Total size: ${(analysis.totalSize / (1024 * 1024)).toFixed(2)} MB`);
        console.log('');
        
        Object.entries(analysis.directories).forEach(([dir, info]) => {
            console.log(`   📁 ${dir}/`);
            console.log(`      Files: ${info.fileCount}`);
            console.log(`      Size: ${info.sizeInMB} MB`);
        });
        
        console.log('\n⚠️  WARNING: This will permanently delete legacy storage directories!');
        console.log('💾 A final backup will be created before deletion.');
        console.log('');
        console.log('To proceed with cleanup, run:');
        console.log('   cleanup.cleanupLegacyFiles(true)');
    }
}

// Usage examples:
const cleanup = new LegacyCleanup();

// Check what would be deleted
console.log('Starting legacy storage cleanup analysis...');
cleanup.interactiveCleanup();

// To actually delete (uncomment when ready):
// cleanup.cleanupLegacyFiles(true);

module.exports = LegacyCleanup;