function movement = detectMove(img1, img2, cases, tform, outputSize, ...
                               percentage_threshold, sensitivityThreshold)

% Redresser les images
gray1 = rgb2gray(img1);
gray2 = rgb2gray(img2);
rectifiedImg1 = imwarp(gray1, tform, 'OutputView', imref2d(outputSize));
rectifiedImg2 = imwarp(gray2, tform, 'OutputView', imref2d(outputSize));

% Calculer la différence
diffImage = imabsdiff(rectifiedImg1, rectifiedImg2);
filteredDiff = diffImage > sensitivityThreshold;

% Identifier les changements de cases
caseNames = fieldnames(cases);
filledBefore = {};
filledAfter = {};

% Supprimer les mouvements parasites avec priorité aux vrais coups
validMoves = struct('from', '', 'to', '');
for c = 1:numel(caseNames)
    % Coordonnées de la case actuelle
    currentCase = cases.(caseNames{c});
    xStart = currentCase.xStart;
    xEnd = currentCase.xEnd;
    yStart = currentCase.yStart;
    yEnd = currentCase.yEnd;

    % Vérifier les différences de pixels dans la case
    currentSquare = filteredDiff(yStart:yEnd, xStart:xEnd);
    totalPixels = numel(currentSquare);
    diffPixels = sum(currentSquare(:));
    percentageDiff = (diffPixels / totalPixels) * 100;

    % Appliquer le seuil de changement significatif
    if percentageDiff > percentage_threshold
        % Comparer les états avant et après
        squareBefore = rectifiedImg1(yStart:yEnd, xStart:xEnd);
        squareAfter = rectifiedImg2(yStart:yEnd, xStart:xEnd);

        meanBefore = mean(squareBefore(:));
        meanAfter = mean(squareAfter(:));

        % Déterminer la présence d'une pièce
        if meanBefore < meanAfter
            filledBefore{end+1} = caseNames{c}; % Pièce retirée
        end
        if meanAfter < meanBefore
            filledAfter{end+1} = caseNames{c}; % Pièce ajoutée
        end
    end
end

% Vérification du coup unique
fromCase = setdiff(filledBefore, filledAfter);
toCase = setdiff(filledAfter, filledBefore);

if numel(fromCase) == 1 && numel(toCase) == 1
    movement = sprintf('%s -> %s', fromCase{1}, toCase{1});
else
    % Priorité aux coups uniques
    if numel(fromCase) == 1
        movement = sprintf('%s -> Inconnu', fromCase{1});
    elseif numel(toCase) == 1
        movement = sprintf('Inconnu -> %s', toCase{1});
    else
        movement = 'Capture ou mouvement multiple détecté';
    end
end

end
