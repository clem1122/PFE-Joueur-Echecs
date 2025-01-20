const chatMessages = document.getElementById('chat-messages');

function appendBotMessage(content) {
    const message = document.createElement('div');
    message.classList.add('message', 'bot');
    message.textContent = content;
    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to the bottom
}

// Example bot messages
const botMessages = [
    "Welcome to the chat!",
    "How can I assist you today?",
    "Here is some information you requested.",
    "Have a great day!",
    "TEST",
    "TEST",
    "TEST",
    "TEST",
    "TEST",
    "TEST",
    "TEST",
    "TEST",
    "TEST"
];

// Simulate messages appearing one by one
let index = 0;
const interval = setInterval(() => {
    if (index < botMessages.length) {
        appendBotMessage(botMessages[index]);
        index++;
    } else {
        clearInterval(interval);
    }
}, 2000);
