const MLServer = require('./api/MLServer');
const { Logger } = require('./utils');

async function main() {
    try {
        Logger.info('🤖 Starting Advanced Trading Bot ML Service...');
        
        // Create and start the ML server
        const server = new MLServer();
        global.tradingBotMLServer = server; // For graceful shutdown
        
        await server.start();
        
        Logger.info('✅ Advanced Trading Bot ML Service started successfully');
        
    } catch (error) {
        Logger.error('❌ Failed to start Advanced Trading Bot ML Service', { 
            error: error.message 
        });
        console.error('Fatal error:', error.message);
        process.exit(1);
    }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
    console.log('\n🛑 Received SIGINT, shutting down ML service gracefully...');
    if (global.tradingBotMLServer) {
        await global.tradingBotMLServer.stop();
    }
    process.exit(0);
});

process.on('SIGTERM', async () => {
    console.log('\n🛑 Received SIGTERM, shutting down ML service gracefully...');
    if (global.tradingBotMLServer) {
        await global.tradingBotMLServer.stop();
    }
    process.exit(0);
});

// Start the application
main();