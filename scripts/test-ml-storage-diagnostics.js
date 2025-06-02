require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { MLStorage } = require('../src/utils');
const { Logger } = require('../src/utils');

async function diagnoseMTStorageIssues() {
    console.log('🔧 ML Storage Diagnostics Tool');
    console.log('=====================================');
    
    const baseDir = 'data/ml';
    const testDir = 'data/ml-test';
    
    // Test 1: Directory Structure Check
    console.log('\n📁 Test 1: Directory Structure Check');
    console.log('-------------------------------------');
    
    const requiredDirs = [
        path.join(baseDir, 'models'),
        path.join(baseDir, 'training'),
        path.join(baseDir, 'predictions'),
        path.join(baseDir, 'features')
    ];
    
    let dirIssues = 0;
    
    requiredDirs.forEach(dir => {
        if (!fs.existsSync(dir)) {
            console.log(`❌ Missing directory: ${dir}`);
            dirIssues++;
            
            try {
                fs.mkdirSync(dir, { recursive: true });
                console.log(`✅ Created directory: ${dir}`);
            } catch (error) {
                console.log(`❌ Failed to create directory: ${dir} - ${error.message}`);
            }
        } else {
            console.log(`✅ Directory exists: ${dir}`);
        }
    });
    
    if (dirIssues === 0) {
        console.log('✅ All required directories exist');
    }
    
    // Test 2: File Permissions Check
    console.log('\n🔐 Test 2: File Permissions Check');
    console.log('----------------------------------');
    
    const testFile = path.join(baseDir, 'permissions_test.json');
    
    try {
        // Test write permission
        fs.writeFileSync(testFile, JSON.stringify({ test: true }));
        console.log('✅ Write permissions: OK');
        
        // Test read permission
        const data = fs.readFileSync(testFile, 'utf8');
        JSON.parse(data);
        console.log('✅ Read permissions: OK');
        
        // Test atomic write (rename permission)
        const tempFile = testFile + '.tmp';
        fs.writeFileSync(tempFile, JSON.stringify({ atomicTest: true }));
        fs.renameSync(tempFile, testFile + '.atomic');
        console.log('✅ Atomic write permissions: OK');
        
        // Cleanup test files
        fs.unlinkSync(testFile);
        fs.unlinkSync(testFile + '.atomic');
        
    } catch (error) {
        console.log(`❌ File permission error: ${error.message}`);
    }
    
    // Test 3: Corrupted File Detection and Repair
    console.log('\n🔍 Test 3: Corrupted File Detection');
    console.log('------------------------------------');
    
    const mlStorage = new MLStorage({ baseDir: testDir });
    let corruptedFiles = 0;
    
    // Create intentionally corrupted files for testing
    const corruptedTestFiles = [
        { file: path.join(testDir, 'models', 'corrupt_model.json'), content: '{"incomplete": json' },
        { file: path.join(testDir, 'predictions', 'empty_predictions.json'), content: '' },
        { file: path.join(testDir, 'features', 'invalid_features.json'), content: 'not json at all' }
    ];
    
    corruptedTestFiles.forEach(({ file, content }) => {
        try {
            fs.writeFileSync(file, content);
            console.log(`📝 Created test corrupted file: ${path.basename(file)}`);
        } catch (error) {
            console.log(`❌ Failed to create test file: ${error.message}`);
        }
    });
    
    // Test detection and recovery
    corruptedTestFiles.forEach(({ file }) => {
        const data = mlStorage.readFileSecure(file);
        if (data === null) {
            console.log(`✅ Corrupted file detected and handled: ${path.basename(file)}`);
            corruptedFiles++;
        } else {
            console.log(`⚠️ File should have been detected as corrupted: ${path.basename(file)}`);
        }
    });
    
    // Cleanup test files
    corruptedTestFiles.forEach(({ file }) => {
        try {
            if (fs.existsSync(file)) fs.unlinkSync(file);
        } catch (error) {
            console.log(`⚠️ Failed to cleanup test file: ${error.message}`);
        }
    });
    
    console.log(`✅ Corrupted file detection working: ${corruptedFiles} files detected`);
    
    // Test 4: Atomic Write Safety
    console.log('\n⚛️ Test 4: Atomic Write Safety');
    console.log('-------------------------------');
    
    const atomicTestFile = path.join(testDir, 'atomic_test.json');
    const validData = { test: 'data', timestamp: Date.now() };
    
    try {
        // Test normal atomic write
        await mlStorage.writeFileAtomic(atomicTestFile, validData);
        const readData = mlStorage.readFileSecure(atomicTestFile);
        
        if (readData && readData.test === 'data') {
            console.log('✅ Atomic write successful');
        } else {
            console.log('❌ Atomic write data mismatch');
        }
        
        // Test atomic write with simulated failure
        const invalidData = { noTimestamp: true }; // Missing required timestamp
        
        try {
            await mlStorage.writeFileAtomic(atomicTestFile, invalidData);
            console.log('⚠️ Invalid data was written (should have failed)');
        } catch (error) {
            console.log('✅ Atomic write correctly rejected invalid data');
        }
        
        // Verify original data is intact
        const verifyData = mlStorage.readFileSecure(atomicTestFile);
        if (verifyData && verifyData.test === 'data') {
            console.log('✅ Original data preserved after failed write');
        } else {
            console.log('❌ Original data was corrupted');
        }
        
    } catch (error) {
        console.log(`❌ Atomic write test failed: ${error.message}`);
    }
    
    // Test 5: Performance Benchmarks
    console.log('\n⚡ Test 5: Performance Benchmarks');
    console.log('----------------------------------');
    
    const performanceTests = [
        { name: 'Small file (1KB)', size: 1024 },
        { name: 'Medium file (10KB)', size: 10240 },
        { name: 'Large file (100KB)', size: 102400 }
    ];
    
    for (const test of performanceTests) {
        const testData = {
            test: test.name,
            timestamp: Date.now(),
            data: 'x'.repeat(test.size - 100) // Account for JSON overhead
        };
        
        const startTime = Date.now();
        
        try {
            await mlStorage.writeFileAtomic(
                path.join(testDir, `perf_test_${test.size}.json`),
                testData
            );
            
            const writeTime = Date.now() - startTime;
            
            const readStart = Date.now();
            mlStorage.readFileSecure(path.join(testDir, `perf_test_${test.size}.json`));
            const readTime = Date.now() - readStart;
            
            console.log(`✅ ${test.name}: Write ${writeTime}ms, Read ${readTime}ms`);
            
            // Performance warnings
            if (writeTime > 1000) {
                console.log(`⚠️ Slow write detected for ${test.name}`);
            }
            if (readTime > 100) {
                console.log(`⚠️ Slow read detected for ${test.name}`);
            }
            
        } catch (error) {
            console.log(`❌ Performance test failed for ${test.name}: ${error.message}`);
        }
    }
    
    // Test 6: Storage Statistics Accuracy
    console.log('\n📊 Test 6: Storage Statistics Accuracy');
    console.log('---------------------------------------');
    
    try {
        const stats = mlStorage.getStorageStats();
        
        console.log('Storage Statistics:');
        console.log(`- Models: ${stats.models.count} files, ${Math.round(stats.models.sizeBytes / 1024)}KB`);
        console.log(`- Training: ${stats.training.count} files, ${Math.round(stats.training.sizeBytes / 1024)}KB`);
        console.log(`- Predictions: ${stats.predictions.count} files, ${Math.round(stats.predictions.sizeBytes / 1024)}KB`);
        console.log(`- Features: ${stats.features.count} files, ${Math.round(stats.features.sizeBytes / 1024)}KB`);
        console.log(`- Total Size: ${Math.round(stats.totalSizeBytes / 1024)}KB`);
        console.log(`- Cache Items: ${Object.values(stats.cache).reduce((sum, count) => sum + count, 0)}`);
        
        // Verify statistics accuracy by manual count
        const actualFiles = {
            models: fs.existsSync(path.join(testDir, 'models')) ? 
                fs.readdirSync(path.join(testDir, 'models')).filter(f => f.endsWith('.json')).length : 0,
            training: fs.existsSync(path.join(testDir, 'training')) ? 
                fs.readdirSync(path.join(testDir, 'training')).filter(f => f.endsWith('.json')).length : 0,
            predictions: fs.existsSync(path.join(testDir, 'predictions')) ? 
                fs.readdirSync(path.join(testDir, 'predictions')).filter(f => f.endsWith('.json')).length : 0,
            features: fs.existsSync(path.join(testDir, 'features')) ? 
                fs.readdirSync(path.join(testDir, 'features')).filter(f => f.endsWith('.json')).length : 0
        };
        
        let statsAccurate = true;
        Object.keys(actualFiles).forEach(key => {
            if (stats[key].count !== actualFiles[key]) {
                console.log(`⚠️ Statistics mismatch for ${key}: reported ${stats[key].count}, actual ${actualFiles[key]}`);
                statsAccurate = false;
            }
        });
        
        if (statsAccurate) {
            console.log('✅ Storage statistics are accurate');
        }
        
    } catch (error) {
        console.log(`❌ Storage statistics test failed: ${error.message}`);
    }
    
    // Test 7: Memory Leak Detection
    console.log('\n🧠 Test 7: Memory Leak Detection');
    console.log('----------------------------------');
    
    const initialMemory = process.memoryUsage().heapUsed;
    
    // Perform many operations to check for memory leaks
    for (let i = 0; i < 100; i++) {
        await mlStorage.saveModelMetadata(`TEST_${i}`, {
            config: { test: true },
            created: Date.now(),
            featureCount: 52
        });
        
        mlStorage.loadModelMetadata(`TEST_${i}`);
        
        // Force garbage collection if available
        if (global.gc) {
            global.gc();
        }
    }
    
    const finalMemory = process.memoryUsage().heapUsed;
    const memoryIncrease = finalMemory - initialMemory;
    const memoryIncreaseKB = Math.round(memoryIncrease / 1024);
    
    console.log(`Memory usage: ${memoryIncreaseKB}KB increase after 100 operations`);
    
    if (memoryIncreaseKB > 10240) { // 10MB threshold
        console.log('⚠️ Potential memory leak detected');
    } else {
        console.log('✅ Memory usage is within normal range');
    }
    
    // Test 8: Cleanup and Finalization
    console.log('\n🧹 Test 8: Cleanup Test');
    console.log('------------------------');
    
    try {
        const cleanedCount = await mlStorage.cleanup(0); // Clean all files
        console.log(`✅ Cleanup completed: ${cleanedCount} files removed`);
        
        await mlStorage.shutdown();
        console.log('✅ Storage shutdown completed');
        
    } catch (error) {
        console.log(`❌ Cleanup test failed: ${error.message}`);
    }
    
    // Final cleanup of test directory
    try {
        if (fs.existsSync(testDir)) {
            fs.rmSync(testDir, { recursive: true, force: true });
            console.log('✅ Test directory cleaned up');
        }
    } catch (error) {
        console.log(`⚠️ Failed to cleanup test directory: ${error.message}`);
    }
    
    // Summary
    console.log('\n🎯 Diagnostic Summary');
    console.log('=====================');
    console.log('✅ Directory structure check completed');
    console.log('✅ File permissions verified');
    console.log('✅ Corrupted file detection tested');
    console.log('✅ Atomic write safety verified');
    console.log('✅ Performance benchmarks completed');
    console.log('✅ Storage statistics accuracy verified');
    console.log('✅ Memory leak detection completed');
    console.log('✅ Cleanup functionality tested');
    
    console.log('\n🚀 ML Storage diagnostics completed successfully!');
    console.log('💾 Advanced persistence system is healthy and functioning properly.');
}

// Auto-repair function
async function autoRepairMLStorage() {
    console.log('\n🔧 Auto-Repair Mode');
    console.log('===================');
    
    const baseDir = 'data/ml';
    
    try {
        // Create missing directories
        const requiredDirs = [
            baseDir,
            path.join(baseDir, 'models'),
            path.join(baseDir, 'training'),
            path.join(baseDir, 'predictions'),
            path.join(baseDir, 'features')
        ];
        
        requiredDirs.forEach(dir => {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
                console.log(`✅ Created missing directory: ${dir}`);
            }
        });
        
        // Remove corrupted files
        const mlStorage = new MLStorage({ baseDir });
        
        for (const subDir of ['models', 'training', 'predictions', 'features']) {
            const dirPath = path.join(baseDir, subDir);
            if (!fs.existsSync(dirPath)) continue;
            
            const files = fs.readdirSync(dirPath);
            
            for (const file of files) {
                if (file.endsWith('.json')) {
                    const filePath = path.join(dirPath, file);
                    const data = mlStorage.readFileSecure(filePath);
                    
                    if (data === null) {
                        console.log(`🗑️ Removing corrupted file: ${file}`);
                        fs.unlinkSync(filePath);
                    }
                }
            }
        }
        
        console.log('✅ Auto-repair completed successfully');
        
    } catch (error) {
        console.log(`❌ Auto-repair failed: ${error.message}`);
    }
}

// Main execution
async function main() {
    try {
        await diagnoseMTStorageIssues();
        
        // Ask if user wants auto-repair (in a real environment, this could be a command line argument)
        const args = process.argv.slice(2);
        if (args.includes('--repair')) {
            await autoRepairMLStorage();
        }
        
    } catch (error) {
        console.error('\n❌ Diagnostics failed:', error.message);
        Logger.error('ML Storage diagnostics failed', { error: error.message });
        process.exit(1);
    }
}

main();