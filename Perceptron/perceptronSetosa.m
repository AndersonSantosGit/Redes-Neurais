function [] =  perceptronSetosa()
    load('Base\iris.mat');
    limiarAtivacao = rand();
    treino = [(ones(size(treino,1),1).*-1),treino];
    label = classes==1;
    taxaApresendizagem = 0.2;
    pesos = [limiarAtivacao,rand(1,2)];
    
    plot(treino(1:50,4),treino(1:50,5),'r*')
    hold on
    plot(treino(51:100,4),treino(51:100,5),'g*')
    plot(treino(101:end,4),treino(101:end,5),'b*')
    
    epocas = 0;
    erro = true;
    while(erro~=false)
        erro = false;
        for i = 1:size(treino,1)
            amostra = [treino(i,1),treino(i,4),treino(i,5)];
            y = sum(amostra.*pesos) >= 0;
            if (label(i)~=y)
                pesos = pesos + (taxaApresendizagem*(label(i)-y)*amostra);
                erro=true;
            end
        end
        epocas=epocas+1;
    end
    epocas
    pesos
    X1 = (0:0.5:7);
    X2 = (pesos(1) - (pesos(2).* X1)) / pesos(3);
    plot(X1,X2);
    
end