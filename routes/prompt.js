const express = require('express');
const { spawn } = require('child_process');
const router = express.Router();
const Prompt = require('../models/Prompt');
const auth = require('../middleware/auth');
const { evaluationQueue } = require('../jobs/evaluationJob');
const User = require('../models/User');  // ✅ Import User model
const multer = require('multer');
const XLSX = require('xlsx');

// ✅ Multer storage configuration
const storage = multer.memoryStorage();
const upload = multer({ storage });

// 📤 Route to handle bulk upload and evaluation
router.post('/evaluate-bulk', auth, upload.single('file'), async (req, res) => {
    try {
        const user_id = req.user.id;
        // ✅ Validate file
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        // ✅ Convert uploaded Excel file to JSON
        const workbook = XLSX.read(req.file.buffer, { type: 'buffer' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const data = XLSX.utils.sheet_to_json(worksheet);

        if (!data.length) {
            return res.status(400).json({ error: 'Uploaded file is empty or invalid.' });
        }

        // ✅ Format prompts for evaluation
        const prompts = data.map(item => ({
            prompt: item.Prompt,
            goldenModel: item.Model
        }));

        // ✅ Save prompts to the database
        const savedPrompts = await Prompt.insertMany(prompts.map(p => ({
            user: user_id,
            prompt: p.prompt,
            goldenModel: p.goldenModel,
            suggestedModel: p.goldenModel,
            isEvaluated: false,
        })));

        // ✅ Update user's evaluation status
        await User.findByIdAndUpdate(user_id, { isEvaluated: false });

        // ✅ Add job to the evaluation queue
        await evaluationQueue.add('evaluate', {
            userId: user_id,
            prompts: prompts
        });

        res.status(200).json({ message: 'File uploaded and evaluation started.' });

    } catch (error) {
        console.error('❌ Error processing uploaded file:', error);
        res.status(500).json({ error: 'Failed to process the uploaded file.' });
    }
});


// Queue the evaluation task
router.post('/evaluate', auth, async (req, res) => {
    try {
        const { prompts } = req.body;  // ✅ Get prompts from request body

        // Debug: Check if prompts are received correctly
        console.log('Received prompts:', prompts);

        // Validate prompts
        if (!prompts || !Array.isArray(prompts) || prompts.length === 0) {
            return res.status(400).json({ error: 'No prompts provided or invalid format' });
        }

        // Add evaluation job to the queue
        await evaluationQueue.add('evaluate', {
            userId: req.user.id,
            prompts: prompts.map(p => ({
                prompt: p.prompt,
                goldenModel: p.goldenModel
            }))
        });

        res.status(200).json({ message: 'Evaluation started. This may take a while.' });
    } catch (error) {
        console.error('Error starting evaluation:', error.message);
        res.status(500).json({ error: 'Failed to start evaluation' });
    }
});


// Check the evaluation status of the logged-in user
router.get('/status', auth, async (req, res) => {
    try {
        // ✅ Find the user and get the isEvaluated field
        const user = await User.findById(req.user.id);

        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }

        res.status(200).json({ isEvaluated: user.isEvaluated });  // ✅ Return isEvaluated status
    } catch (error) {
        console.error('Error checking evaluation status:', error.message);
        res.status(500).json({ error: 'Failed to check status' });
    }
});

// DELETE /api/prompts/:id - Delete a prompt by ID
router.delete('/:id', auth, async (req, res) => {
    try {
        const prompt = await Prompt.findById(req.params.id);

        if (!prompt) {
            return res.status(404).json({ error: 'Prompt not found' });
        }

        // Ensure only the owner can delete the prompt
        if (prompt.user.toString() !== req.user.id) {
            return res.status(401).json({ error: 'Unauthorized to delete this prompt' });
        }

        // 🗑️ Delete related Comparisons
        await Comparison.deleteMany({ prompt: prompt._id });

        // 🗑️ Delete related GoldStandard
        if (prompt.goldStandard) {
            await GoldStandard.findByIdAndDelete(prompt.goldStandard);
        }

        // 🗑️ Delete related Decision
        if (prompt.decision) {
            await Decision.findByIdAndDelete(prompt.decision);
        }

        // Use findByIdAndDelete instead of remove
        await Prompt.findByIdAndDelete(req.params.id);

        res.json({ message: 'Prompt deleted successfully' });

    } catch (err) {
        console.error('Error deleting prompt:', err.message);
        res.status(500).json({ error: 'Server Error' });
    }
});

// Get prompts for logged-in user
router.get('/', auth, async (req, res) => {
    try {
        const prompts = await Prompt.find({ user: req.user.id });
        res.json(prompts);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

router.get('/results', auth, async (req, res) => {
    try {
        const prompts = await Prompt.find({ user: req.user.id})
            .populate('goldStandard')
            .populate('comparisons')
            .populate('decision');

        res.status(200).json(prompts);
    } catch (error) {
        console.error('Error fetching results:', error);
        res.status(500).json({ error: 'Failed to fetch evaluation results' });
    }
});

// Add a new prompt
router.post('/', auth, async (req, res) => {
    const { prompt, goldenModel, suggestedModel } = req.body;

    try {
        // ✅ Create a new prompt
        const newPrompt = new Prompt({
            user: req.user.id,
            prompt: prompt,
            goldenModel: goldenModel,
            suggestedModel: suggestedModel || goldenModel,
        });

        await newPrompt.save();

        // ✅ Add the prompt reference to the User's prompts array
        await User.findByIdAndUpdate(req.user.id, {
            $push: { prompts: newPrompt._id }
        });

        res.status(201).json(newPrompt);
    } catch (err) {
        console.error('Error adding prompt:', err.message);
        res.status(500).json({ error: err.message });
    }
});



module.exports = router;
