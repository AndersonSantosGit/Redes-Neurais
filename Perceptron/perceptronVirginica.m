function [] =  perceptronVirginica()
    load('Base\iris.mat');
    limiarAtivacao = rand(); 
    treino = [(ones(size(treino,1),1).*-1),treino];
    label = classes==3;
    taxaApresendizagem = 0.05;
    pesos = [limiarAtivacao,rand(1,2)];
    
    plot(treino(1:100,4),treino(1:100,5),'g*')
    hold on
    plot(treino(101:end,4),treino(101:end,5),'r*')
    
    %treino = [ [treino(1:50,1);treino(101:150,1)] , [treino(1:50,4);treino(101:150,4)] , [treino(1:50,5);treino(101:150,5)] ];
    %label = [label(1:50);label(101:150)];

    epocas = 0;
    erroTotal = +Inf;
    while(erroTotal>2)
        erroTotal = 0;
        for i = 1:size(treino,1)          
            amostra = [treino(i,1),treino(i,2),treino(i,3)];
            y = sum(amostra.*pesos) >= 0;
            erro = label(i) - y;
            erroTotal = erroTotal + (erro^2);
            if (erro~=0)
                pesos = pesos + (taxaApresendizagem*amostra*(label(i)-y));
            end
        end
        epocas = epocas+1;
    end
    X1 = (0:0.5:7);
    X2 = (pesos(1) - (pesos(2).* X1)) / pesos(3);
    plot(X1,X2); 
end