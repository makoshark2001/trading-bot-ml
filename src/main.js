const MLServer = require('./api/MLServer');
const { Logger } = require('./utils');

async function main() {
    try {
        Logger.info('🤖 Starting Advanced Trading Bot ML Service with Enhanced Persistence...');
        
        // Create and start the ML server
        const server = new MLServer();
        global.tradingBotMLServer = server; // For graceful shutdown
        
        await server.start();
        
        Logger.info('✅ Advanced Trading Bot ML Service started successfully');
        Logger.info('💾 Enhanced persistence enabled with atomic writes and caching');
        
    } catch (error) {
        Logger.error('❌ Failed to start Advanced Trading Bot ML Service', { 
            error: error.message 
        });
        console.error('Fatal error:', error.message);
        process.exit(1);
    }
}

// Enhanced graceful shutdown with storage persistence
async function gracefulShutdown(signal) {
    console.log(`\n🛑 Received ${signal}, shutting down ML service gracefully...`);
    Logger.info(`Graceful shutdown initiated by ${signal}`);
    
    if (global.tradingBotMLServer) {
        try {
            // Stop the server which will trigger storage shutdown
            await global.tradingBotMLServer.stop();
            Logger.info('✅ ML service shutdown completed successfully');
            console.log('💾 All ML data saved with atomic writes');
        } catch (error) {
            Logger.error('❌ Error during graceful shutdown', { 
                error: error.message 
            });
            console.error('Shutdown error:', error.message);
        }
    }
    
    // Give a moment for final log writes
    setTimeout(() => {
        process.exit(0);
    }, 1000);
}

// Handle graceful shutdown signals
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

// Handle uncaught exceptions with storage cleanup
process.on('uncaughtException', async (error) => {
    console.error('\n💥 Uncaught Exception:', error.message);
    Logger.error('Uncaught exception occurred', { 
        error: error.message,
        stack: error.stack 
    });
    
    // Try to save data before exiting
    if (global.tradingBotMLServer) {
        try {
            console.log('🚨 Attempting emergency data save...');
            await global.tradingBotMLServer.stop();
            console.log('💾 Emergency save completed');
        } catch (saveError) {
            console.error('❌ Emergency save failed:', saveError.message);
        }
    }
    
    process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', async (reason, promise) => {
    console.error('\n⚠️ Unhandled Rejection at:', promise, 'reason:', reason);
    Logger.error('Unhandled promise rejection', { 
        reason: reason?.message || reason,
        stack: reason?.stack 
    });
    
    // For unhandled rejections, log but don't exit immediately
    // The application might recover
});

// Handle process warnings
process.on('warning', (warning) => {
    Logger.warn('Process warning', {
        name: warning.name,
        message: warning.message,
        stack: warning.stack
    });
});

// Memory usage monitoring
setInterval(() => {
    const usage = process.memoryUsage();
    const memoryMB = Math.round(usage.heapUsed / 1024 / 1024);
    
    // Log memory usage if it's getting high (over 1GB)
    if (memoryMB > 1024) {
        Logger.warn('High memory usage detected', {
            heapUsedMB: memoryMB,
            heapTotalMB: Math.round(usage.heapTotal / 1024 / 1024),
            externalMB: Math.round(usage.external / 1024 / 1024),
            rss: Math.round(usage.rss / 1024 / 1024)
        });
    }
}, 300000); // Check every 5 minutes

// Startup banner with consolidated storage info
console.log(`
╔══════════════════════════════════════════════════════════════╗
║                   🤖 TRADING BOT ML SERVICE                  ║
║              4-Model Ensemble + Consolidated Storage         ║
╠══════════════════════════════════════════════════════════════╣
║  🧠 LSTM + GRU + CNN + Transformer  💾 Consolidated Storage ║
║  📊 84+ Feature Engineering         🔄 Training Queue Mgmt  ║
║  🔮 Real-time Ensemble Predictions  📈 Atomic File Writes   ║
║  ⚡ Intelligent Caching            🧹 Automatic Migration   ║
║  🚀 Ultra-Fast Performance         📊 Enhanced Analytics    ║
╠══════════════════════════════════════════════════════════════╣
║  Port: 3001                         Version: 2.0.0          ║
║  Node: ${process.version.padEnd(22)} Status: Starting...           ║
╚══════════════════════════════════════════════════════════════╝
`);

// Start the application
main();