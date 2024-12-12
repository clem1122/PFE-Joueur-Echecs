clear all
close all
clc

% 0. Parametres
percentage_threshold = 22; % Threshold poucentage d'erreur
sensitivityThreshold = 40; % Sensibilité pour les différences

% 1. Charger l'image de référence (échiquier vide)
referenceImagePath = 'empty.png';
referenceImage = imread(referenceImagePath);
referenceGray = rgb2gray(referenceImage);

% 2. Sélection manuelle des coins (à faire une seule fois)
figure;
imshow(referenceGray);
title('Cliquez sur les 4 coins de l''échiquier (ordre : haut-gauche, haut-droit, bas-droit, bas-gauche)');
[x, y] = ginput(4); % Sélection des 4 coins manuellement
inputPoints = [x, y]; % Points sélectionnés
close;

% 3. Paramètres pour le redressement et la détection
outputSize = [800, 800]; % Taille du redressement (800x800 pixels)
squareSize = outputSize(1) / 8; % Taille d'une case en pixels
outputPoints = [1, 1; outputSize(2), 1; outputSize(2), outputSize(1); 1, outputSize(1)];

% 4. Redressement de l'image de référence
tform = fitgeotrans(inputPoints, outputPoints, 'projective');
rectifiedReference = imwarp(referenceImage, tform, 'OutputView', imref2d(outputSize));

% 5. Calculer et stocker les coordonnées des cases
cases = struct(); % Structure pour stocker les coordonnées
for row = 1:8
    for col = 1:8
        % Calculer les coordonnées des cases une fois
        xStart = round((col - 1) * squareSize) + 1;
        xEnd = round(col * squareSize);
        yStart = round((8 - row) * squareSize) + 1; % Ligne inversée pour suivre la convention
        yEnd = round((8 - row + 1) * squareSize);

        % Nom de la case (e.g., A1, B2)
        caseName = [char('A' + col - 1), num2str(row)];

        % Stocker les coordonnées dans une structure
        cases.(caseName) = struct('xStart', xStart, 'xEnd', xEnd, ...
            'yStart', yStart, 'yEnd', yEnd);
    end
end

%%

% 6. Charger toutes les images à analyser
imageFolder = 'photos';
imageFiles = dir(fullfile(imageFolder, '*.png')); % Récupérer toutes les images PNG

% 7. MAIN LOOP
for i = 1:(length(imageFiles)-1)
    % Charger les images consécutives
    img1Path = fullfile(imageFolder, imageFiles(i).name);
    img2Path = fullfile(imageFolder, imageFiles(i+1).name);
    img1 = imread(img1Path);
    img2 = imread(img2Path);

    % Convertir en niveaux de gris
    gray1 = rgb2gray(img1);
    gray2 = rgb2gray(img2);

    % Redresser les images avec les mêmes points
    rectifiedImg1 = imwarp(gray1, tform, 'OutputView', imref2d(outputSize));
    rectifiedImg2 = imwarp(gray2, tform, 'OutputView', imref2d(outputSize));

    % Calculer la différence entre les deux images redressées
    diffImage = imabsdiff(rectifiedImg1, rectifiedImg2);
    filteredDiff = diffImage > sensitivityThreshold; % Appliquer le seuil

    % Identifier les cases contenant des différences
    fprintf('Differences between %s & %s :\n', imageFiles(i).name, imageFiles(i+1).name);
    caseNames = fieldnames(cases); % Récupérer tous les noms des cases
    for c = 1:numel(caseNames)
        % Récupérer les coordonnées de la case
        currentCase = cases.(caseNames{c});
        xStart = currentCase.xStart;
        xEnd = currentCase.xEnd;
        yStart = currentCase.yStart;
        yEnd = currentCase.yEnd;

        % Vérifier les différences dans cette case
        currentSquare = filteredDiff(yStart:yEnd, xStart:xEnd);
        %totalPixels = numel(currentSquare); % Nombre total de pixels dans la case
        totalPixels = squareSize^2;
        diffPixels = sum(currentSquare(:)); % Nombre de pixels présentant une différence
        percentageDiff = round((diffPixels / totalPixels) * 100); % Pourcentage de différence

        if percentageDiff > percentage_threshold % Si des différences sont présentes
            fprintf(' - %s: %.2f%%\n', caseNames{c}, percentageDiff);
        end
    end

    % Superposer la différence avec l'image de référence redressée
    figure;
    imshow(rectifiedReference); % Affiche l'échiquier vide redressé
    hold on;

    % Superposer les différences avec transparence
    h = imshow(filteredDiff);
    set(h, 'AlphaData', 0.5); % Transparence

    % Ajouter la grille des cases
    for row = 1:8
        % Lignes horizontales
        yLine = round((8 - row) * squareSize); % Position de la ligne
        plot([1, outputSize(2)], [yLine, yLine], 'w-', 'LineWidth', 1); % Ligne blanche
    end
    for col = 1:8
        % Lignes verticales
        xLine = round((col - 1) * squareSize); % Position de la colonne
        plot([xLine, xLine], [1, outputSize(1)], 'w-', 'LineWidth', 1); % Ligne blanche
    end

    % Ajouter les labels des cases
    for row = 1:8
        for col = 1:8
            caseName = [char('A' + col - 1), num2str(row)]; % Nom de la case (e.g., A1, B2)
            xText = round((col - 1) * squareSize) + squareSize / 2; % Position X
            yText = round((8 - row) * squareSize) + squareSize / 2; % Position Y
            text(xText, yText, caseName, 'Color', 'yellow', 'FontSize', 10, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        end
    end
    title(['Differences between ', imageFiles(i).name, '&', imageFiles(i+1).name]);
    hold off;
end
