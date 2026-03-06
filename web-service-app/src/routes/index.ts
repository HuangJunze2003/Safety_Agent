import express from 'express';
import axios from 'axios';

const router = express.Router();

export function setRoutes(app: express.Express) {
    app.use('/', router);

    router.post('/api/chat', async (req, res) => {
        try {
            const { message, imageBase64 } = req.body;
            let content: any[] = [];
            
            if (imageBase64) {
                content.push({ type: "image_url", image_url: { url: imageBase64 } });
            }
            if (message) {
                content.push({ type: "text", text: message });
            } else {
                content.push({ type: "text", text: "请分析这张图" });
            }

            const payload = {
                model: "qwen2-vl",
                messages: [{ role: "user", content }],
                max_tokens: 1024,
                temperature: 0.1
            };

            const response = await axios.post('http://127.0.0.1:8000/v1/chat/completions', payload);
            const reply = response.data.choices[0].message.content;
            res.json({ reply });
        } catch (error: any) {
            console.error("API call error:", error?.message);
            res.status(500).json({ error: error?.message || "Service error" });
        }
    });
}
