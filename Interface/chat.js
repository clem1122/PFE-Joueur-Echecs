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

const Messages = {
    'checked': "Attention, tu es en echec, ton roi est attaqué !",
    'checkmated': "J'ai gagné ! Tu t'es bien défendu, mais ton roi ne peut plus esquiver cette attaque",
    'check': "Joli ! Tu attaques mon roi.",
    'checkmate': "Tu as gagné ! Je m'incline, mon roi est définitivement perdu...",
    'threats': ""

}

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
