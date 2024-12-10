    async function getInfo(toggleId) {
      try {
        const response = await fetch(`http://127.0.0.1:5000/get-info/${toggleId}`);
        const data = await response.json();

        if (response.ok) {
          alert(`Infos pour ${toggleId}: ${data.info}`);
        } else {
          alert(`Erreur: ${data.error}`);
        }
      } catch (error) {
        alert("Impossible de récupérer les infos.");
        console.error(error);
      }
    }