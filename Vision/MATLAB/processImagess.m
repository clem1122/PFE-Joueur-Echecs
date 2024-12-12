function processImages(img1, img2, rectifiedReference, tform, cases, ...
    sensitivityThreshold, percentage_threshold, ...
    squareSize, outputSize, img1Name, img2Name)

% Fonction pour calculer les différences entre deux images et les afficher

% 1. Redresser les images en utilisant l'image de calibration
gray1 = rgb2gray(img1);
gray2 = rgb2gray(img2);
rectifiedImg1 = imwarp(gray1, tform, 'OutputView', imref2d(outputSize));
rectifiedImg2 = imwarp(gray2, tform, 'OutputView', imref2d(outputSize));

% 2. Calculer les différences
diffImage = imabsdiff(rectifiedImg1, rectifiedImg2);
filteredDiff = diffImage > sensitivityThreshold;

% 3. Identifier les cases contenant des différences
fprintf('Differences between %s & %s :\n', img1Name, img2Name);
caseNames = fieldnames(cases);
modifiedCases = {}; % Stocker les cases modifiées

for c = 1:numel(caseNames)
    % Récupérer les coordonnées de la case
    currentCase = cases.(caseNames{c});
    xStart = currentCase.xStart;
    xEnd = currentCase.xEnd;
    yStart = currentCase.yStart;
    yEnd = currentCase.yEnd;

    % Vérifier les différences
    currentSquare = filteredDiff(yStart:yEnd, xStart:xEnd);
    totalPixels = squareSize^2;
    diffPixels = sum(currentSquare(:));
    percentageDiff = round((diffPixels / totalPixels) * 100);

    if percentageDiff > percentage_threshold
        modifiedCases{end+1} = caseNames{c}; %#ok<AGROW>
        fprintf(' - %s: %.2f%%\n', caseNames{c}, percentageDiff);
    end
end

% 4. Afficher les résultats
figure;
imshow(rectifiedReference);
hold on;

% Superposer les différences
h = imshow(filteredDiff);
set(h, 'AlphaData', 0.5);

% Dessiner la grille
for row = 1:8
    yLine = round((8 - row) * squareSize);
    plot([1, outputSize(2)], [yLine, yLine], 'w-', 'LineWidth', 1);
end
for col = 1:8
    xLine = round((col - 1) * squareSize);
    plot([xLine, xLine], [1, outputSize(1)], 'w-', 'LineWidth', 1);
end

% Ajouter les labels
for row = 1:8
    for col = 1:8
        caseName = [char('A' + col - 1), num2str(row)];
        xText = round((col - 1) * squareSize) + squareSize / 2;
        yText = round((8 - row) * squareSize) + squareSize / 2;
        text(xText, yText, caseName, 'Color', 'yellow', 'FontSize', 10, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
end

title(['Differences between ', img1Name, ' & ', img2Name]);
hold off;
end
