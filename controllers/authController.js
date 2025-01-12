const User = require('../models/User');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');



// Register Controller
const register = async (req, res) => {
    const { username, email, password } = req.body;
    try {
        if (!email) {
            return res.status(400).json({ error: 'Email is required.' });
        } else if (!username) {
            return res.status(400).json({ error: 'Username is required.' });
        } else if (!password) {
            return res.status(400).json({ error: 'Password is required.' });
        }

        const newUser = new User({ username, email, password });
        await newUser.save();
        res.status(201).json({ message: 'Successfully Registered' });
    } catch (err) {
        if (err.code === 11000) {
            const field = Object.keys(err.keyValue)[0];
            const value = err.keyValue[field];
            res.status(400).json({ error: `The ${field} "${value}" is already in use. Please choose a different ${field}.` });
        } else {
            res.status(400).json({ error: err.message });
        }
    }
};

// Login Controller
const login = async (req, res) => {
    const { email, password } = req.body;
    try {
        if (!email || !password) {
            return res.status(400).json({ error: 'Email and password are required.' });
        }

        const user = await User.findOne({ email });
        if (!user) {
            return res.status(400).json({ error: 'User not found. Please register first.' });
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(400).json({ error: 'Incorrect password. Please try again.' });
        }

        const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });

        // Send token and isEvaluated status
        res.status(200).json({
            token,
            isEvaluated: user.isEvaluated,  // ✅ Send isEvaluated to frontend
            message: 'Logged in successfully',
        });
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
};

// Profile Controller
const profile = async (req, res) => {
    try {
        const user = await User.findById(req.user.id).select('-password');
        res.json(user);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
};

module.exports = { register, login, profile };
