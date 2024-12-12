clear
close all
clc

% 0. Paramètres
percentage_threshold = 22; % Seuil pour le pourcentage de différence
sensitivityThreshold = 40; % Sensibilité pour les différences
outputSize = [800, 800]; % Taille du redressement (800x800 pixels)
squareSize = outputSize(1) / 8; % Taille d'une case en pixels
calibrationFile = 'chessboard_calibration.mat'; % Fichier pour sauvegarder les coins calibrés

% 1. Calibration des coins
if isfile(calibrationFile)
    % Charger la calibration existante si le fichier existe deja
    load(calibrationFile, 'inputPoints');
    disp('Calibration depuis fichier.');
else
    % Si le fichier n'existe pas, on fait la calibration sur l'image vide
    referenceImagePath = 'empty.png';
    referenceImage = imread(referenceImagePath);
    referenceGray = rgb2gray(referenceImage);

    % Sélection manuelle des coins
    figure;
    imshow(referenceGray);
    title('Cliquez sur les 4 coins de l''échiquier (haut-gauche, haut-droit, bas-droit, bas-gauche)');
    [x, y] = ginput(4); % Sélection des coins
    inputPoints = [x, y]; % Points sélectionnés
    close;

    % Sauvegarder le fichier de calibration
    save(calibrationFile, 'inputPoints');
    disp('Calibration des coins enregistrée.');
end

% 2. Redresser l'image de référence
outputPoints = [1, 1; outputSize(2), 1; outputSize(2), outputSize(1); 1, outputSize(1)];
tform = fitgeotform2d(inputPoints, outputPoints, 'projective');

% Charger l'image de référence (échiquier vide) pour superposer 
referenceImagePath = 'empty.png';
referenceImage = imread(referenceImagePath);
rectifiedReference = imwarp(referenceImage, tform, 'OutputView', imref2d(outputSize));

% 3. Calculer et stocker les coordonnées des cases
cases = struct();
for row = 1:8
    for col = 1:8
        xStart = round((col - 1) * squareSize) + 1;
        xEnd = round(col * squareSize);
        yStart = round((8 - row) * squareSize) + 1;
        yEnd = round((8 - row + 1) * squareSize);
        caseName = [char('A' + col - 1), num2str(row)];
        cases.(caseName) = struct('xStart', xStart, 'xEnd', xEnd, ...
                                  'yStart', yStart, 'yEnd', yEnd);
    end
end

% 4. Charger les images à analyser
imageFolder = 'photos';
imageFiles = dir(fullfile(imageFolder, '*.png'));

% 5. Boucle principale
for i = 1:(length(imageFiles)-1)
    % Charger deux images consécutives
    img1Path = fullfile(imageFolder, imageFiles(i).name);
    img2Path = fullfile(imageFolder, imageFiles(i+1).name);
    img1 = imread(img1Path);
    img2 = imread(img2Path);

    % Appeler la fonction pour analyser les différences
    processImages(img1, img2, rectifiedReference, tform, cases, ...
        sensitivityThreshold, percentage_threshold, squareSize, outputSize, ...
        imageFiles(i).name, imageFiles(i+1).name);
end
