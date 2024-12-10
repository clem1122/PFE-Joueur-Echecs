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
let isWhite = true;
let Box;


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
    drawBoard();
}

function drawBoard() {
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let x = j * squareSize;
            let y = i * squareSize;
            fill(colors[(i + j) % 2]);
            if (selectedSquare != null) {
		        let {row, col} = selectedSquare;
		        if(row == i && col == j) {
			        
		        	fill(selectedColor)
		        }
            }
            rect(x, y, squareSize, squareSize);
            
        }
    }
    drawPieces();
}

function drawPieces(){
	for (let i = 0; i < rows; i++) {
		for (let j = 0; j < cols; j++) {
				let FENindex = isWhite ? i * 8 + j : 63 - (i * 8 + j);
				if (FEN[FENindex] != '.') {
					let x = j * squareSize;
		           	let y = i * squareSize;
					image(pieceImages[FEN[FENindex]], x, y);
				}
		    }
		}
}
 




function mousePressed() {


	
	let _col = floor(mouseX / squareSize);
	let _row = floor(mouseY / squareSize);

	if (_col >= 0 && _col < 8 && _row >= 0 && _row < 8) {
	
		let FENindex = isWhite ? _row * 8 + _col : 63 - (_row * 8 + _col);
		if(selectedPiece == null) {
			if (FEN[FENindex] != '.') {
				selectedPiece = FEN[FENindex];
				console.log(selectedPiece);
			}
			selectedSquare = { row: _row, col: _col };
		} else {
		
			
			let endCoord = isWhite ? String.fromCharCode(97 + _col) + (8 - _row) : String.fromCharCode(104 - _col) + (1+_row);
			
			let {row, col} = selectedSquare;
			
			let startCoord = isWhite ? String.fromCharCode(97 + col) + (8 - row) : String.fromCharCode(104 - col) + (1+row);
			let move = startCoord + endCoord;
			
			selectedPiece = null;
			selectedSquare = null;
			sendData(move);
			
		}
		
		
		
		
    	drawBoard();
	}
}








