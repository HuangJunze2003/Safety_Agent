let currentImageBase64 = null;

function appendMessage(sender, text, b64Image=null) {
    const chatBox = document.getElementById('chatBox');
    const div = document.createElement('div');
    div.className = `msg ${sender}`;
    div.innerText = sender === 'user' ? "你: " + text : "智能体: " + text;
    
    if (b64Image) {
        const img = document.createElement('img');
        img.src = b64Image;
        div.appendChild(img);
    }
    
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            currentImageBase64 = e.target.result;
            const preview = document.getElementById('imagePreview');
            preview.src = currentImageBase64;
            document.getElementById('previewContainer').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

function clearImage() {
    currentImageBase64 = null;
    document.getElementById('imageInput').value = '';
    document.getElementById('previewContainer').style.display = 'none';
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message && !currentImageBase64) return;

    appendMessage('user', message, currentImageBase64);
    input.value = '';
    
    const requestData = { message, imageBase64: currentImageBase64 };
    clearImage(); // reset

    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'msg bot';
    loadingDiv.innerText = "智能体思考中...";
    document.getElementById('chatBox').appendChild(loadingDiv);

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        const data = await res.json();
        const chatBox = document.getElementById('chatBox');
        chatBox.removeChild(loadingDiv);
        
        if (data.reply) {
            appendMessage('bot', data.reply);
        } else {
            appendMessage('bot', `[错误] ${data.error}`);
        }
    } catch (e) {
        document.getElementById('chatBox').removeChild(loadingDiv);
        appendMessage('bot', `[网络错误] ${e.message}`);
    }
}

function handleEnter(e) {
    if (e.key === 'Enter') sendMessage();
}
