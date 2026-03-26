import express from 'express';
import axios from 'axios';
import multer from 'multer';
import FormData from 'form-data';
import fs from 'fs';

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

export function setRoutes(app: express.Express) {
    app.use('/', router);

    router.post('/api/chat', async (req, res) => {
        // ... (existing chat route)
    });

    router.post('/api/upload', upload.single('file'), async (req, res) => {
        try {
            const { libType } = req.body;
            const file = req.file;

            if (!file) {
                return res.status(400).json({ error: 'No file uploaded' });
            }

            // 转发给 Python 处理 API (8001端口)
            const formData = new FormData();
            formData.append('file', fs.createReadStream(file.path), file.originalname);
            formData.append('lib_type', libType);

            const response = await axios.post('http://127.0.0.1:8001/upload', formData, {
                headers: {
                    ...formData.getHeaders(),
                },
            });

            // 清理本地临时文件
            fs.unlinkSync(file.path);

            res.json(response.data);
        } catch (error: any) {
            console.error("Upload error:", error?.message);
            res.status(500).json({ error: error?.message || "Internal server error" });
        }
    });
}

