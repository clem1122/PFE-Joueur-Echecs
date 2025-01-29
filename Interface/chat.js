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

function appendMessage(content, sender) {
    const message = document.createElement('div');
    message.classList.add('message', sender);
    message.textContent = content;
    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to the bottom
}

// Example bot messages
const StartMessages = [
    "Bonjour ! Je suis Niryo ! Je suis un robot pédagogique pour enseigner les échecs. On joue ? 😊"
    //"Pour information, les pièces déjà présentes dans le cimetière en début de partie sont simplement prévues en cas de promotion."
];

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
    'unsure' : "J'ai détecté un coup de ta part qui n'est pas légal : ",

}

// Simulate messages appearing one by one
let index = 0;
let previous_state = {
    "check": false,
    "checkmate": false,
    "checked": false,
    "checkmated": false,
    "unsure" : "",
}

let previous_FEN_to_show = {'threats': false, 'controlled': false, 'playable': false, 'help': false, 'protected':false};

setInterval(fetchAndAppendMessages, 1000);
setInterval(getMessage, 1000);

const interval = setInterval(() => {
    if (index < StartMessages.length) {
        appendMessage(StartMessages[index], 'bot');
        index++;
    } else {
        clearInterval(interval);
    }
}, 2000);

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
                console.log(state[cle])
                if (state[cle] || FEN_to_show[cle]){
                    console.log(cle)
                    console.log(message)
                    if (cle == "unsure"){
                        console.log("Robot unsure")
                        message += state[cle]
                        message += ". Ecris-moi le coup que tu souhaitais jouer."
                    }

                    appendBotMessage(message);
                    break
                }
                i++;
                }

            // if (state['checkmated']) {
            //     appendBotMessage(Messages['checkmated']);
            // } else if (state['checked']) {
            //     appendBotMessage(Messages['checked']);
            // } else if (state['checkmate']) {
            //     appendBotMessage(Messages['checkmate']);
            // } else if (state['check']) {
            //     appendBotMessage(Messages['check']);
            // } else if (FEN_to_show['threats']) {
            //     appendBotMessage(Messages['threats']);
            // } else if (FEN_to_show['playable']) {
            //     appendBotMessage(Messages['playable']);
            // } else if (FEN_to_show['controlled']) {
            //     appendBotMessage(Messages['controlled']);
            // } else if (FEN_to_show['protected']) {
            //     appendBotMessage(Messages['protected']);
            // } else if (FEN_to_show['aide']) {
            //     appendBotMessage(Messages['aide']);
            // }

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