async function getInfo(toggleId) {

  const buttonIds = ['threats', 'controlled', 'playable']; // Liste des IDs connus
  const FEN_to_show = {'threats': false, 'controlled': false, 'playable': false};
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
