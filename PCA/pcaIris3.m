function [output] = pcaIris3()

    %%  Step 1 - Obter os dados
    load('..\Perceptron\Base\iris.mat');
    treino = treino(:,2:4);
    [m,n] = size(treino);
    k = 2;
    
    %% Plotando os dados 
    figure(1);
    hold on;
    h = gscatter(treino(:,1),treino(:,2),classes);
    % for each unique group in 'g', set the ZData property appropriately
    gu = unique(classes);
    z = treino(:,3);
    for p = 1:numel(gu)
          set(h(p), 'ZData', z( classes == gu(p) ));
    end
    view(3);
    
    %%  Step 2 - Subtrair a média
    media = mean(treino,1);
    dataAjustada = treino;
    dataAjustada(:,1) = treino(:,1)-media(1);
    dataAjustada(:,2) = treino(:,2)-media(2);
    dataAjustada(:,3) = treino(:,3)-media(3);
    
    %% Plotando os dados ajustados
    figure(2);
    hold on;
    h = gscatter(dataAjustada(:,1),dataAjustada(:,2),classes);
    % for each unique group in 'g', set the ZData property appropriately
    gu = unique(classes);
    z = dataAjustada(:,3);
    for p = 1:numel(gu)
          set(h(p), 'ZData', z( classes == gu(p) ));
    end
    view(3);

    %% Step 3 - Calcular a matriz de covariancia
    matrizCov = cov(dataAjustada);
    %matrizCov2 = (1/(m-1))*(dataAjustada'*dataAjustada);
    
    %% Step 4 - Autovetores e autovalores
    [autovetores, autovalores] = eig(matrizCov);
    diagonal = diag(autovalores);
    [~,I] = sort(diagonal,'descend');
    
    %% Step 5 - Derivando o nova base
    output = autovetores(:,I(1:k,1))'*dataAjustada';
    figure(3);
    hold on;
    gscatter(output(1,:),output(2,:),classes);
    figure(4);
    hold on;
    gscatter(output(1,:),zeros(size(output(1,:))),classes);
end