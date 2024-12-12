clear all
close all
clc

% Spécifiez le dossier contenant les images
imageFolder = 'photos';

referenceImage = imread("empty.png");
referenceGray = rgb2gray(referenceImage); % Conversion en niveaux de gris

% Récupérez la liste des fichiers dans le dossier
imageFiles = dir(fullfile(imageFolder, '*.png')); % Changez '*.png' en fonction de votre format

% Paramètre de sensibilité (seuil d'intensité pour les différences)
sensitivityThreshold = 30; % Ajustez entre 0 et 255 (plus grand = moins sensible)

% Parcourir les images deux à deux
for i = 1:(length(imageFiles)-1)
    % Charger les images consécutives
    img1 = imread(fullfile(imageFolder, imageFiles(i).name));
    img2 = imread(fullfile(imageFolder, imageFiles(i+1).name));

    % Conversion en niveaux de gris
    gray1 = rgb2gray(img1);
    gray2 = rgb2gray(img2);

    % Calcul de la différence brute
    diffImage = imabsdiff(gray1, gray2);

    % Appliquer le seuil de sensibilité
    filteredDiff = diffImage > sensitivityThreshold;

    % Superposer la différence sur l'image de référence
    figure;
    imshow(referenceImage); % Afficher l'image de référence
    hold on;
    h = imshow(filteredDiff); % Superposer les différences détectées
    set(h, 'AlphaData', 0.5); % Transparence pour voir la référence en arrière-plan
    title(['Différence détectée (seuil = ', num2str(sensitivityThreshold), ...
           ') : Image ', num2str(i), ' et ', num2str(i+1)]);
    hold off;
end
