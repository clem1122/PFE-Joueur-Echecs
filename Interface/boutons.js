async function getInfo(toggleId) {

  const buttonIds = ['threats', 'controlled', 'playable','help','protected']; // Liste des IDs connus
  const FEN_to_show = {'threats': false, 'controlled': false, 'playable': false, 'help': false, 'protected':false};
  const all_toggle_buttons = buttonIds.map(id => document.getElementById(id));

  try {
    const response = await fetch(`http://127.0.0.1:5000/get-info/${toggleId}`);
    const data = await response.json();
    const toggle_button = document.getElementById(`${toggleId}`)

    if (response.ok) {
      if (toggle_button.checked){
        FEN_to_show[`${toggleId}`] = true;
        console.log("sending : " + JSON.stringify(FEN_to_show));
        localStorage.setItem("FEN_to_show", JSON.stringify(FEN_to_show));

        all_toggle_buttons.forEach(button => {
          if (button.id !== toggleId) {
            button.checked = false; // Uncheck other buttons
          }
        });
      }
      else {
        FEN_to_show[`${toggleId}`] = false;
        localStorage.setItem("FEN_to_show", JSON.stringify(FEN_to_show));
      }

    } else {
      alert(`Erreur: ${data.error}`);
    }
  } catch (error) {
    alert("Impossible de récupérer les infos.");
    console.error(error);
  }
}


async function have_played(){

console.log("J'ai joué !");

try {
  // Envoi d'une requête POST au serveur Flask
  const response = await fetch('http://127.0.0.1:5000/set-have-played', 
  {
    method: 'POST',
    headers: {'Content-Type': 'application/json',},
    body: JSON.stringify({ have_played: true }), // Envoi d'un booléen
  });

  if (response.ok) {
    console.log('Données de have_played envoyées avec succès depuis le .js !');
  } else {
    console.error('Erreur lors de l\'envoi des données de have_played depuis le .js.');
  }

} catch (error) {console.error('Erreur réseau :', error);}
}