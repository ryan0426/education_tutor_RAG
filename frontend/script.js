async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    if (!message) return;
  
    appendMessage('You', message, 'user');
    input.value = '';
  
    appendMessage('Tutor', 'Typing...', 'bot');
  
    try {
      const res = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
  
      const data = await res.json();
  
      // Remove "Typing..." message
      removeLastMessage();
  
      appendMessage('Tutor', data.response, 'bot');
    } catch (err) {
      removeLastMessage();
      appendMessage('Tutor', '⚠️ Error connecting to backend.', 'bot');
      console.error(err);
    }
  }
  
  function appendMessage(sender, text, cls) {
    const chatBox = document.getElementById('chat-box');
    const message = document.createElement('div');
    message.className = `message ${cls}`;
    message.innerText = `${sender}: ${text}`;
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
  }
  
  function removeLastMessage() {
    const chatBox = document.getElementById('chat-box');
    if (chatBox.lastChild) {
      chatBox.removeChild(chatBox.lastChild);
    }
  }
  