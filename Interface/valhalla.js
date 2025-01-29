function preload() {
    pieceImages['Q'] = loadImage("Images/Q.png");
    pieceImages['K'] = loadImage("Images/K.png");
    pieceImages['P'] = loadImage("Images/P.png");
    pieceImages['N'] = loadImage("Images/N.png");
    pieceImages['B'] = loadImage("Images/B.png");
    pieceImages['R'] = loadImage("Images/R.png");

    pieceImages['q'] = loadImage("Images/q.png");
    pieceImages['k'] = loadImage("Images/k.png");
    pieceImages['p'] = loadImage("Images/p.png");
    pieceImages['n'] = loadImage("Images/n.png");
    pieceImages['b'] = loadImage("Images/b.png");
    pieceImages['r'] = loadImage("Images/r.png");
}

<<<<<<< HEAD
function setup(){
    const canvas = createCanvas(700, 700);
    canvas.parent('valhalla');
}
=======
function setup() {
    // Crée un canevas pour l'échiquier

    // Crée un canevas distinct pour chaque cimetière
    const whiteCemeteryCanvas = createGraphics(300, 150);
    whiteCemeteryCanvas.parent('valhalla-white');

    const blackCemeteryCanvas = createGraphics(300, 150);
    blackCemeteryCanvas.parent('valhalla-black');
}
>>>>>>> a8992b6 (presque ça)
