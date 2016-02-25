function [output] = pcaIris()

    %%  Step 1 - Obter os dados
    load('..\Perceptron\Base\iris.mat');
    treino = treino(:,3:4);
    [m,n] = size(treino);
    k = 1;
    figure(1);
    hold on;
    gscatter(treino(:,1),treino(:,2),classes);
    
    %%  Step 2 - Subtrair a média
    media = mean(treino,1);
    dataAjustada = treino;
    dataAjustada(:,1) = treino(:,1)-media(1);
    dataAjustada(:,2) = treino(:,2)-media(2);
    figure(2);
    hold on;
    gscatter(dataAjustada(:,1),dataAjustada(:,2),classes);

    %% Step 3 - Calcular a matriz de covariancia
    %matrizCov = cov(dataAjustada(:,1),dataAjustada(:,2));
    matrizCov = cov(dataAjustada);
    %matrizCov2 = (1/m)*(	'*dataAjustada);
    
    %% Step 4 - Autovetores e autovalores
    [autovetores, autovalores] = eig(matrizCov);
    diagonal = diag(autovalores);
    [~,I] = sort(diagonal,'descend');
    
    %% Step 5 - Derivando o nova base
    output = autovetores(:,I(1:k,1))'*dataAjustada';
    figure(3);
    hold on;
    gscatter(output,zeros(size(output(1,:))),classes);
end