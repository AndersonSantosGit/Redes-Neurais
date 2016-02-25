function GA_MLP()
    tic
    n_intermediaria = 94;
    %% ETAPA DE PRÉ-PROCESSAMENTO
    load('..\PMC\DataSet\Training\Train.mat');
    dataTraining = train(1:1000,2:end);
    trainingLabel = getLabels(train(1:1000,1));
    clear('train');
    
    %trainingData =  f_features;
    Fitness_Threshold = size(dataTraining,1)*0.2;
    p = 100;
    r = 0.6;
    contador_Geracoes = 1;
    m = 0.4;
    %% Initialize population
    %P = rand(p,74646); %  W1 -> P(:,1:76382) W2 -> P(:,76383:84682)
    P = randn(p,74646); %  W1 -> P(:,1:76382) W2 -> P(:,76383:84682)
    
    %EVALUATE
    aptidoes = zeros(size(P,1),1);
    for i = 1:size(P,1)
        h = P(i,:);
        [w1,w2] = convertValues(h);    
        aptidoes(i) = getAptidaoMpcDigitRecognizer(w1,w2,dataTraining,trainingLabel,n_intermediaria);
    end
    
    while min(aptidoes) > Fitness_Threshold
        
        fprintf ('Geração %d, Aptidão: %d  \n', contador_Geracoes, min(aptidoes));
        %quantidadePs = floor(1-r)*p;
        quantidadeCross = floor(r*p);
        [~,In] = sort(aptidoes);
        PSort = P(In,:);
        dataCrossOver = PSort(1:quantidadeCross,:);
        i = 1;
        while i < quantidadeCross
            pai1 = dataCrossOver(i,:);
            pai2 = dataCrossOver(i+1,:);
            [PSort(101-i,:),PSort(101-(i+1),:)] = crossingOver(pai1,pai2);
            i = i+2;
        end
        %OPERAÇÃO DE MUTAÇÃO
        if m > rand()
            P = mutacao(PSort);
        else
            P = PSort;
        end
        %EVALUATE
        for i = 1:size(P,1)
            h = P(i,:);
            [w1,w2] = convertValues(h);
            aptidoes(i) = getAptidaoMpcDigitRecognizer(w1,w2,dataTraining,trainingLabel,n_intermediaria);
        end
        %save(['Results\optimization_geracao_', num2str(contador_Geracoes), '.mat'], 'optimization');
        contador_Geracoes = contador_Geracoes + 1;
    end
    
    for i = 1:size(P,1)
        h = P(i,:);
        [w1,w2] = convertValues(h);
        aptidoes(i) = getAptidaoMpcDigitRecognizer(w1,w2,trainingLabel,label);
    end
    save('Results\optimization_geracao_final.mat', 'w1','w2','P','aptidoes');
    toc
    
end

function data = mutacao(data)
    n = rand(74646,1);
    indicePadrao = randi(10);
    indiceMutacao = find(n==max(n));
    data(indicePadrao,indiceMutacao) = data(indicePadrao,indiceMutacao) + randn();
end

function [filho1 ,filho2] = crossingOver(pai1, pai2)
    
    indice_corte = randi(74646, 1, 1);
    filho1 = [pai1(1:indice_corte), mean( [ pai1(indice_corte+1:end); pai2(indice_corte+1:end)])];
    filho2 = [pai2(1:indice_corte), mean( [ pai1(indice_corte+1:end); pai2(indice_corte+1:end)])];
    
end

function [training, validation] = getData(data)
    
    data = shuffle(data,2);
    total = size(data, 2);
    indice_corte = floor(0.70 * total);
    
    training = data(1:indice_corte);
    validation = data(indice_corte+1:total);
    
end

function [w1,w2] = convertValues(hipotese)

    w1 = reshape(hipotese(1:73696),94,784);
    w2 = reshape(hipotese(73697:end),10,95);
    
end

function [output] = getLabels(data)
    output = zeros(size(data,1),10);
    for k =1 :size(data,1)
        switch data(k,1)
            case 0
                output(k,:) = [1,0,0,0,0,0,0,0,0,0];
            case 1
                output(k,:) = [0,1,0,0,0,0,0,0,0,0];
            case 2
                output(k,:) = [0,0,1,0,0,0,0,0,0,0];
            case 3
                output(k,:) = [0,0,0,1,0,0,0,0,0,0];
            case 4
                output(k,:) = [0,0,0,0,1,0,0,0,0,0]; 
            case 5
                output(k,:) = [0,0,0,0,0,1,0,0,0,0];
            case 6
                output(k,:) = [0,0,0,0,0,0,1,0,0,0];
            case 7
                output(k,:) = [0,0,0,0,0,0,0,1,0,0];
            case 8
                output(k,:) = [0,0,0,0,0,0,0,0,1,0];
            case 9
                output(k,:) = [0,0,0,0,0,0,0,0,0,1];                
        end
    end
end