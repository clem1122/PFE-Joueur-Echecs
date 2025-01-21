const chatMessages = document.getElementById('chat-messages');

function appendBotMessage(content) {
    const message = document.createElement('div');
    message.classList.add('message', 'bot');
    message.textContent = content;
    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to the bottom
}

// Example bot messages
const StartMessages = [
    "Bonjour ! Je suis Niryo",
    "Je suis un robot pédagogique pour enseigner les échecs",
    "Tu veux jouer une partie avec moi ?",
    "😊😊😊"
];

const Messages = {
    'checked': "Attention, tu es en echec, ton roi est attaqué !",
    'checkmated': "J'ai gagné ! Tu t'es bien défendu, mais ton roi ne peut plus esquiver cette attaque",
    'check': "Joli ! Tu attaques mon roi.",
    'checkmate': "Tu as gagné ! Je m'incline, mon roi est définitivement perdu...",
    'threats': "Regarde, voici les pièces que je te menace.",
    'playable': "Voila toutes les cases que tu peux atteindre avec une de tes pièces",
    'controlled': "Prudence sur ces cases, je les controlle avec une de mes pièces.",
    'protected': "Ces pièces sont protégées : si tu les captures, je pourrais te capturer derrière.",
    'aide': "Si tu veux mon avis, le meilleur coup pour toi est "
}

// Simulate messages appearing one by one
let index = 0;
const interval = setInterval(() => {
    if (index < StartMessages.length) {
        appendBotMessage(StartMessages[index]);
        index++;
    } else {
        clearInterval(interval);
    }
}, 2000);
