let rows = 8;
let cols = 8;
let squareSize;
let colors = ["#f3dbb4", "#b38c62"];
let selectedColor = "#98c47e";
let FEN = "rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR";
let pieceImages = {};
let selectedSquare = null;
let selectedPiece = null;
let port = "8080";
let url = "http://localhost:" + port;
let isFlipped = false; // Boolean for board orientation, false for normal, true for 180° rotated
let Box;
let threats = '.'.repeat(64);
let controlled = '.'.repeat(64);
let playable = '.'.repeat(64);
let help = '.'.repeat(64);
localStorage.setItem("threats", threats);
localStorage.setItem("controlled", controlled);
localStorage.setItem("playable", playable);
localStorage.setItem("help", help);

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

function setup() {
    const canvas = createCanvas(500, 500);
    canvas.parent('chessboard');
    squareSize = width / (rows + 2); // Adjust size to leave space for labels
    for (let key in pieceImages) {
        pieceImages[key].resize(squareSize, 0);
    }

    noStroke();
    frameRate(10);
}

function draw() {
    FEN_to_show = JSON.parse(localStorage.getItem("FEN_to_show"));

    if (frameCount % 10 == 0) {
        updateFENs();
    }
    drawBoardWithLabels();
    if (FEN_to_show['threats']) { draw_color_FEN(threats, color(150, 0, 0, 175)); }
    if (FEN_to_show['controlled']) { draw_color_FEN(controlled, color(0, 0, 150, 100)); }
    if (FEN_to_show['playable']) { draw_color_FEN(playable, color(0, 150, 0, 100)); }
    if (FEN_to_show['help']) { draw_color_FEN(help, color(165, 32, 100, 170)); }
    drawPieces();
}

async function updateFENs() {
    getBoardFEN();
    color_FEN = await getColorFEN();
    if (color_FEN !== undefined) {
        threats = color_FEN['threats'];
        controlled = color_FEN['controlled'];
        playable = color_FEN['playable'];
        help = color_FEN['help']
    }
}

function draw_color_FEN(FEN, couleur) {
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let x = (isFlipped ? cols - j : j + 1) * squareSize;
            let y = (isFlipped ? rows - i : i + 1) * squareSize;
            fill(couleur);
            if (FEN[i * 8 + j] == '1') {
                rect(x, y, squareSize, squareSize);
            }
        }
    }
}

function drawBoardWithLabels() {
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let x = (isFlipped ? cols - j : j + 1) * squareSize;
            let y = (isFlipped ? rows - i : i + 1) * squareSize;
            fill(colors[(i + j) % 2]);
            if (selectedSquare != null) {
                let { row, col } = selectedSquare;
                if (row == i && col == j) {
                    fill(selectedColor);
                }
            }
            rect(x, y, squareSize, squareSize);
        }
    }

    // Draw labels
    fill(0);
    textSize(squareSize * 0.4);
    textAlign(CENTER, CENTER);
    for (let i = rows-1; i >=0  ; i--) {
        let label = isFlipped ? i + 1 : rows - i;
        text(label, 0.5 * squareSize, (i + 1.5) * squareSize);
        text(label, (cols + 1.5) * squareSize, (i + 1.5) * squareSize);
    }
    for (let j = 0; j < cols; j++) {
        let label = isFlipped ? String.fromCharCode(104 - j) : String.fromCharCode(97 + j);
        text(label, (j + 1.5) * squareSize, 0.5 * squareSize);
        text(label, (j + 1.5) * squareSize, (rows + 1.5) * squareSize);
    }
}

function drawPieces() {
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let FENindex = i * 8 + j;
            if (FEN[FENindex] != '.') {
                let x = (isFlipped ? cols - j : j + 1) * squareSize;
                let y = (isFlipped ? rows - i : i + 1) * squareSize;
                image(pieceImages[FEN[FENindex]], x, y);
            }
        }
    }
}

function mousePressed() {
    let _col = floor(mouseX / squareSize) - 1;
    let _row = floor(mouseY / squareSize) - 1;

    if (_col >= 0 && _col < 8 && _row >= 0 && _row < 8) {
        let adjustedCol = isFlipped ? cols - 1 - _col : _col;
        let adjustedRow = isFlipped ? rows - 1 - _row : _row;
        let FENindex = adjustedRow * 8 + adjustedCol;

        if (selectedPiece == null) {
            if (FEN[FENindex] != '.') {
                selectedPiece = FEN[FENindex];
            }
            selectedSquare = { row: adjustedRow, col: adjustedCol };
        } else {
            let endCoord = String.fromCharCode(97 + adjustedCol) + (8 - adjustedRow);

            let { row, col } = selectedSquare;
            let startCoord = String.fromCharCode(97 + col) + (8 - row);
            let move = startCoord + endCoord;

            selectedPiece = null;
            selectedSquare = null;
        }

        drawBoardWithLabels();
    }
}

async function getBoardFEN() {
    const url_board = "http://127.0.0.1:5000/get-board-FEN";
    const response = await fetch(url_board);
    const data = await response.json();
    FEN = data.board_FEN;
}

async function getColorFEN() {
    const url_color_FEN = "http://127.0.0.1:5000/get-color-FEN";
    const response = await fetch(url_color_FEN);
    const color_FEN = await response.json();
    return color_FEN;
}
