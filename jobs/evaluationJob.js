const { Worker, Queue } = require('bullmq');
const { spawn } = require('child_process');
const mongoose = require('mongoose');
const Prompt = require('../models/Prompt');
const Comparison = require('../models/Comparison');
const GoldStandard = require('../models/GoldStandard');
const Decision = require('../models/Decision');
const User = require('../models/User');

// ✅ Updated Redis connection to port 6380
require('dotenv').config();

// ✅ Updated Redis connection using environment variable
const connection = {
    connection: {
        url: process.env.REDIS_URL  // 🔑 Use Internal Redis URL
    }
};
// ✅ Create a queue for evaluation jobs
const evaluationQueue = new Queue('evaluation', { connection });

// ✅ Worker to process evaluation jobs
const evaluationWorker = new Worker('evaluation', async (job) => {
    console.log(`🚀 Processing job for user ${job.data.userId}`);

    try {
        // ✅ Mark user's evaluation as started
        await User.findByIdAndUpdate(job.data.userId, { isEvaluated: false });

        // ✅ Spawn Python process
        const pythonProcess = spawn('python', ['main.py']);

        // ✅ Send data to Python script
        pythonProcess.stdin.write(JSON.stringify(job.data.prompts));
        pythonProcess.stdin.end();

        let result = '';

        // ✅ Capture Python output
        pythonProcess.stdout.on('data', (data) => {
            const output = data.toString();

            // ✅ Check if the output is JSON
            if (output.trim().startsWith('{') || output.trim().startsWith('[')) {
                result += output;  // ✅ Append JSON content
            } else {
                console.log(`📥 Python Debug Output: ${output}`);  // Log non-JSON debug output
            }
        });

        // ✅ Capture Python errors
        pythonProcess.stderr.on('data', (data) => {
            console.error(`❌ Python Error: ${data.toString()}`);
        });

        // ✅ Handle Python process close
        pythonProcess.on('close', async code => {
            if (code === 0) {
                let parsedResult;

                try {
                    // ✅ Check if result is valid
                    if (!result) {
                        throw new Error('Python script returned no data');
                    }

                    parsedResult = JSON.parse(result);
                } catch (err) {
                    console.error(`❌ Failed to parse Python output: ${err.message}`);
                    return;  // ⛔ Stop further processing if parsing failed
                }

                // ✅ Continue processing if parsedResult is valid
                for (const [promptText, evaluation] of Object.entries(parsedResult)) {
                    const prompt = await Prompt.findOne({ prompt: promptText, user: job.data.userId });

                    if (!prompt) {
                        console.warn(`⚠️ Prompt not found for text: ${promptText}`);
                        continue;
                    }

                    // ✅ Save Gold Standard
                    const goldStandard = new GoldStandard({
                        model: evaluation['Gold Standard'].Model,
                        latency: evaluation['Gold Standard']['Latency (ms)'],
                        tokenEfficiencyRatio: evaluation['Gold Standard']['Token Efficiency Ratio'],
                        accuracy: evaluation['Gold Standard'].Accuracy,
                        response: evaluation['Gold Standard'].Response
                    });

                    try {
                        await goldStandard.save();
                    } catch (error) {
                        console.error(`❌ Error saving GoldStandard for prompt "${promptText}": ${error.message}`);
                    }

                    // ✅ Save Comparisons
                    const comparisons = evaluation.Comparisons.length > 0
                        ? await Promise.all(
                            evaluation.Comparisons.map(async (comp) => {
                                const comparison = new Comparison({
                                    prompt: prompt._id,
                                    model: comp.Model,
                                    metrics: {
                                        latency: comp.Metrics['Latency (ms)'],
                                        tokenEconomyGain: comp.Metrics['Token Economy Gain (%)'],
                                        semanticSimilarity: comp.Metrics['Semantic Similarity'],
                                        coherence: comp.Metrics['Coherence'],
                                        accuracy: comp.Metrics['Accuracy'],
                                        response: comp.Metrics['Response']
                                    },
                                    verdict: comp.Verdict,
                                    reasoning: comp.Reasoning
                                });
                                return await comparison.save();
                            })
                        )
                        : [];


                    // ✅ Save Final Decision
                    const decision = new Decision({
                        bestModel: evaluation['Final Decision']['Best Model'],
                        preferredForLatency: evaluation['Final Decision']['Preferred for Latency'],
                        preferredForTokenEfficiency: evaluation['Final Decision']['Preferred for Token Efficiency']
                    });
                    await decision.save();

                    // ✅ Update Prompt with references
                    prompt.goldStandard = goldStandard._id;
                    prompt.comparisons = comparisons.map(c => c._id);
                    prompt.decision = decision._id;
                    await prompt.save();

                    console.log(`✅ Prompt "${promptText}" evaluated and updated.`);
                }

                await User.findByIdAndUpdate(job.data.userId, { isEvaluated: true });

                console.log(`🎉 Evaluation for user ${job.data.userId} completed.`);
            } else {
                console.error(`❌ Python process exited with code ${code}`);
            }
        });


    } catch (error) {
        console.error(`❌ Error processing job for user ${job.data.userId}:`, error);
    }
}, { connection });

module.exports = { evaluationQueue };
