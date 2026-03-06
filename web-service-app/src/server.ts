import express from 'express';
import cors from 'cors';
import { setRoutes } from './routes/index.js';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({limit: '50mb'}));
app.use(express.static('public'));

setRoutes(app);

app.listen(PORT, () => {
    console.log(`Frontend running on http://localhost:${PORT}`);
});
