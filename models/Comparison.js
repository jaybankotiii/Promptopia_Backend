const mongoose = require('mongoose');

const ComparisonSchema = new mongoose.Schema({
    prompt: { type: mongoose.Schema.Types.ObjectId, ref: 'Prompt', required: true },
    model: { type: String, required: true },
    metrics: {
        latency: { type: Number, required: true },
        tokenEconomyGain: { type: Number, required: true },
        semanticSimilarity: { type: Number, required: true },
        coherence: { type: Number, required: true },
        accuracy: { type: Number, required: true },
        response: { type: String, required: true }
    },
    verdict: { type: String, required: true },
    reasoning: { type: String, required: true }
}, { timestamps: true });

module.exports = mongoose.model('Comparison', ComparisonSchema);
