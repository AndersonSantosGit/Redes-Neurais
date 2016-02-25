function [] = mpcDigitRecognizer()
    tic
    n_intermediaria = 94;
    %% ETAPA DE PRÉ-PROCESSAMENTO
    load('DataSet\Training\Train.mat');
    dataTraining = train(:,2:end);
    trainingLabel = getLabels(train(:,1));
    clear('train');
    
    [w_1,w_2,erros] = getMpcDigitRecognizer(dataTraining,trainingLabel,n_intermediaria);
    %load('VARS.mat');
    %disp(erros);
    clear('dataTraining','trainingLabel');
    %plot(1:3001,erros);
    
    %Entradas para os neurônios
    I_1 = zeros(n_intermediaria,1);
    I_2 = zeros(10,1);
    
    %Saida dos Neurônios
    Y_1 = ones(n_intermediaria+1,1).*-1;
    Y_2 = ones(10,1).*-1;
    
    load('DataSet\Testing\DataTest.mat');
    outputs = zeros(size(test,1),10);
    %Obter Y_2 ajustado
    for a = 1:size(test,1)
        amostra = test(a,:);
        amostra = mat2gray(amostra);
        %Passo Forward
        for n1 = 1:size(I_1,1)
            I_1(n1) = sum(amostra.*w_1(n1,:));
            Y_1(n1+1) = sigmf( I_1(n1),[0.2 10] );
        end
        for n2 = 1:size(I_2,1)
            I_2(n2) = sum(w_2(n2,:).*Y_1');
            Y_2(n2) = sigmf( I_2(n2),[0.2 10] );
        end
        SAIDA = zeros(1,10);
        SAIDA(max(Y_2)==Y_2)= 1;
        outputs(a,:) = SAIDA;
    end 
   
    result = convertOutputs(outputs);
    ids = [1:28000]';
    labels = {'ImageId','Label'};
    data = [ids,result];
    csvwrite('OUTPUT.csv',data,labels);

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

function [results] = convertOutputs( outputs)

    results = zeros(size(outputs,1),1);
    for k =1 :size(outputs,1)
        padrao = outputs(k,:);
        number = find(padrao == 1);
        if (size(number,2)==1)
           results(k) = number-1;
        else
           results(k) = number(randi(size(number,2)))-1;
        end
    end
    
end