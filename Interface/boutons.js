async function getInfo(toggleId) {

  const buttonIds = ['threat_button', 'control_button', 'play_button']; // Liste des IDs connus
  const all_toggle_buttons = buttonIds.map(id => document.getElementById(id));
  
  try {
    const response = await fetch(`http://127.0.0.1:5000/get-info/${toggleId}`);
    const data = await response.json();
    const toggle_button = document.getElementById(`${toggleId}`)

    if (response.ok) {
      if (toggle_button.checked){
        localStorage.setItem(`${toggleId}`, data.FEN);
        all_toggle_buttons.forEach(button => {
          if (button.id !== toggleId) {
            button.checked = false; // Uncheck other buttons
            localStorage.setItem(button.id, '.'.repeat(64)); // Update their storage state
          }
        });
      }
      else{
        localStorage.setItem(`${toggleId}`, '.'.repeat(64));
      }

    } else {
      alert(`Erreur: ${data.error}`);
    }
  } catch (error) {
    alert("Impossible de récupérer les infos.");
    console.error(error);
  }
}


