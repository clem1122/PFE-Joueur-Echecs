const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');

sendButton.addEventListener('click', () => {
    const userMessage = chatInput.value.trim();
    if (userMessage) {
        appendMessage(userMessage, 'user');
        chatInput.value = '';
        fetch("http://127.0.0.1:5000/set-answer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({reponse: userMessage})
        })
    }
});

chatInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        sendButton.click();
    }
});

function appendMessage(content, sender, type = null) {
    const message = document.createElement('div');
    message.classList.add('message', sender);
    message.textContent = content;
    chatMessages.appendChild(message);
    message.classList.add('message', sender);

    if (type) {
        message.classList.add(type); // Ajoute la classe correspondant au type de message
    }

    message.textContent = content;
    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll automatique vers le bas
}




const Messages = {
    'checkmate': "Tu as gagné ! Je m'incline, mon roi est définitivement perdu...",
    'checkmated': "J'ai gagné ! Tu t'es bien défendu(e), mais ton roi ne peut plus esquiver cette attaque",
    'checked': "Attention, tu es en echec, ton roi est attaqué !",
    'check': "Joli ! Tu attaques mon roi.",
    'queen_threat': "Prend garde à ta reine, elle est menacée",
    'threats': "Regarde, voici les pièces que je te menace.",
    'playable': "Voila toutes les cases que tu peux atteindre avec une de tes pièces",
    'controlled': "Prudence sur ces cases, je les controle avec une de mes pièces.",
    'protected': "Ces pièces sont protégées : si je les captures, tu pourras me capturer derrière.",
    'aide': "Si tu veux mon avis, le meilleur coup pour toi est de faire celui-ci.",
}

// Simulate messages appearing one by one
let index = 0;
let previous_state = {
    "check": false,
    "checkmate": false,
    "checked": false,
    "checkmated": false
}

let previous_FEN_to_show = {'threats': false, 'controlled': false, 'playable': false, 'help': false, 'protected':false};

setInterval(fetchAndAppendMessages, 1000);
setInterval(getMessage, 1000);


async function fetchAndAppendMessages() {
    try {
        const url_state = "http://127.0.0.1:5000/get-state";
        const response = await fetch(url_state);
        FEN_to_show = JSON.parse(localStorage.getItem("FEN_to_show"));

        if (!response.ok) {
            console.error("Failed to fetch state:", response.statusText);
            return;
        }
        const state = await response.json();
        console.log("state : ", JSON.stringify(state));
        console.log("previous_state : ", JSON.stringify(previous_state));

        const keys = Object.entries(Messages);
        let i = 0;

        if (JSON.stringify(previous_state) !== JSON.stringify(state) || JSON.stringify(previous_FEN_to_show) !== JSON.stringify(FEN_to_show)) {
            while (i < keys.length) {
                let [cle, message] = keys[i];
                if (state[cle] || FEN_to_show[cle]){
                    console.log(cle, message);
                    appendMessage(message, 'bot', cle); // Passer `cle` comme type
                    break;
                }
                i++;
            }

            previous_state = state;
            previous_FEN_to_show = FEN_to_show;
        }

    } catch (error) {
        console.error("Error fetching messages:", error);
    }
}

async function getState() {
    const url_state = "http://127.0.0.1:5000/get-state";
    const response = await fetch(url_state);
    const state = await response.json();
    return state;
}

async function getMessage() {
    const url_state = "http://127.0.0.1:5000/get-message";
    const response = await fetch(url_state);
    const msg = await response.json();
    if(msg['message'] != "") {
        appendMessage(msg['message'], 'bot');
    }
}