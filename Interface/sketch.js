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
let isFlipped = true; // Boolean for board orientation, false for normal, true for 180° rotated
let Box;
let threats = '.'.repeat(64);
let controlled = '.'.repeat(64);
let playable = '.'.repeat(64);
localStorage.setItem("threats", threats);
localStorage.setItem("controlled", controlled);
localStorage.setItem("playable", playable);

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
    const canvas = createCanvas(600, 600);
    canvas.parent('chessboard');
    squareSize = width / rows;
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
    drawBoard();
    if (FEN_to_show['threats']) { draw_color_FEN(threats, color(150, 0, 0, 175)); }
    if (FEN_to_show['controlled']) { draw_color_FEN(controlled, color(0, 0, 150, 100)); }
    if (FEN_to_show['playable']) { draw_color_FEN(playable, color(0, 150, 0, 100)); }
    drawPieces();
}

async function updateFENs() {
    getBoardFEN();
    color_FEN = await getColorFEN();
    if (color_FEN !== undefined) {
        threats = color_FEN['threats'];
        controlled = color_FEN['controlled'];
        playable = color_FEN['playable'];
    }
}

function draw_color_FEN(FEN, couleur) {
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let x = (isFlipped ? cols - 1 - j : j) * squareSize;
            let y = (isFlipped ? rows - 1 - i : i) * squareSize;
            fill(couleur);
            if (FEN[i * 8 + j] == '1') {
                rect(x, y, squareSize, squareSize);
            }
        }
    }
}

function drawBoard() {
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let x = (isFlipped ? cols - 1 - j : j) * squareSize;
            let y = (isFlipped ? rows - 1 - i : i) * squareSize;
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
}

function drawPieces() {
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let FENindex = i * 8 + j;
            if (FEN[FENindex] != '.') {
                let x = (isFlipped ? cols - 1 - j : j) * squareSize;
                let y = (isFlipped ? rows - 1 - i : i) * squareSize;
                image(pieceImages[FEN[FENindex]], x, y);
            }
        }
    }
}

function mousePressed() {
    let _col = floor(mouseX / squareSize);
    let _row = floor(mouseY / squareSize);

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

        drawBoard();
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
