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

function setup(){
    const canvas = createCanvas(700, 700);
    canvas.parent('valhalla');
}