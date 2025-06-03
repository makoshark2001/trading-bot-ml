require('dotenv').config();
const fs = require('fs');
const path = require('path');

async function cleanMLStorage() {
    console.log('🧹 Cleaning ML Storage (Model Files)...');
    console.log('======================================');
    
    const mlDataDir = path.join(process.cwd(), 'data', 'ml');
    const modelsDir = path.join(mlDataDir, 'models');
    
    try {
        // Check if directories exist
        if (!fs.existsSync(mlDataDir)) {
            console.log('❌ ML data directory does not exist:', mlDataDir);
            return;
        }
        
        if (!fs.existsSync(modelsDir)) {
            console.log('❌ Models directory does not exist:', modelsDir);
            return;
        }
        
        // List current model files
        console.log('\n📊 Step 1: Current Model Files...');
        const modelFiles = fs.readdirSync(modelsDir);
        console.log('Found model files:', modelFiles);
        
        // Delete model metadata files
        console.log('\n📊 Step 2: Deleting Model Metadata Files...');
        let deletedCount = 0;
        
        for (const file of modelFiles) {
            if (file.endsWith('_model.json')) {
                const filePath = path.join(modelsDir, file);
                try {
                    fs.unlinkSync(filePath);
                    console.log(`✅ Deleted: ${file}`);
                    deletedCount++;
                } catch (error) {
                    console.log(`❌ Failed to delete ${file}:`, error.message);
                }
            }
        }
        
        console.log(`\n✅ Deleted ${deletedCount} model metadata files`);
        
        // Check training directory
        const trainingDir = path.join(mlDataDir, 'training');
        if (fs.existsSync(trainingDir)) {
            console.log('\n📊 Step 3: Cleaning Training History...');
            const trainingFiles = fs.readdirSync(trainingDir);
            let trainingDeleted = 0;
            
            for (const file of trainingFiles) {
                if (file.endsWith('_training.json')) {
                    const filePath = path.join(trainingDir, file);
                    try {
                        fs.unlinkSync(filePath);
                        console.log(`✅ Deleted training: ${file}`);
                        trainingDeleted++;
                    } catch (error) {
                        console.log(`❌ Failed to delete training ${file}:`, error.message);
                    }
                }
            }
            
            console.log(`✅ Deleted ${trainingDeleted} training history files`);
        }
        
        // Check features cache
        const featuresDir = path.join(mlDataDir, 'features');
        if (fs.existsSync(featuresDir)) {
            console.log('\n📊 Step 4: Clearing Feature Cache...');
            const featureFiles = fs.readdirSync(featuresDir);
            let featuresDeleted = 0;
            
            for (const file of featureFiles) {
                if (file.endsWith('_features.json')) {
                    const filePath = path.join(featuresDir, file);
                    try {
                        fs.unlinkSync(filePath);
                        console.log(`✅ Deleted features: ${file}`);
                        featuresDeleted++;
                    } catch (error) {
                        console.log(`❌ Failed to delete features ${file}:`, error.message);
                    }
                }
            }
            
            console.log(`✅ Deleted ${featuresDeleted} feature cache files`);
        }
        
        // Verify cleanup
        console.log('\n📊 Step 5: Verifying Cleanup...');
        const remainingModelFiles = fs.readdirSync(modelsDir).filter(f => f.endsWith('.json'));
        console.log('Remaining model files:', remainingModelFiles.length ? remainingModelFiles : 'None');
        
        console.log('\n🎉 ML Storage Cleanup Complete!');
        console.log('===============================');
        console.log('✅ All model metadata files deleted');
        console.log('✅ Training history cleared');
        console.log('✅ Feature cache cleared');
        console.log('');
        console.log('🚀 Next Steps:');
        console.log('1. Restart the ML service: npm start');
        console.log('2. Models will be rebuilt with correct feature count (84)');
        console.log('3. Test predictions should work without feature count errors');
        console.log('');
        console.log('💡 The service will automatically detect the current feature count');
        console.log('   and build all models with 84 features instead of 52.');
        
    } catch (error) {
        console.error('\n❌ Storage cleanup failed:', error.message);
        process.exit(1);
    }
}

cleanMLStorage();