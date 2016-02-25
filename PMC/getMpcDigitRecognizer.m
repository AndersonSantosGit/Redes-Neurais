function [w_1,w_2,erros] = getMpcDigitRecognizer(f_features,label,n_intermediaria)
tic
    %% TREINAMENTO DA MCP
    %Vetores de pesos
    w_1 = rand(n_intermediaria,784);
    w_2 = rand(10,n_intermediaria+1);
    
    %Entradas para os neurônios
    I_1 = zeros(n_intermediaria,1);
    I_2 = zeros(10,1);
    
    %Saida dos Neurônios
    Y_1 = ones(n_intermediaria+1,1).*-1;
    Y_2 = ones(10,1).*-1;
    
    %AJUSTE DE PARÂMETROS
    taxaApresendizagem = 0.5;
    outputs = zeros(size(f_features,1),10);
    erros = [];
    erroAtual = E(label,outputs);
    epocas = 1;
    %for epocas = 1:100
    while erroAtual>0
        tic
        disp(['Ep >>> ', num2str(epocas)]);
        erros = [erros,erroAtual];
        disp(num2str(erroAtual));
        for a = 1:size(f_features,1)
            amostra = f_features(a,:);
            amostra = mat2gray(amostra);
            %Passo Forward
            for n1 = 1:size(I_1,1)
                I_1(n1) = sum(amostra.*w_1(n1,:));
                Y_1(n1+1) = sigmf(I_1(n1),[0.2 10]);
            end
            for n2 = 1:size(I_2,1)
                I_2(n2) = sum(w_2(n2,:).*Y_1');
                Y_2(n2) = sigmf(I_2(n2),[0.2 10]);
            end
            %Passo Backward
            delta_2 = (label(a,:) - Y_2') .* ( (1-sigmf(I_2',[0.2 10])) .* (sigmf(I_2',[0.2 10])) );
            for j = 1:size(delta_2',1)
                for i = 1:size(Y_1,1)
                    w_2(j,i) = w_2(j,i) + (taxaApresendizagem*delta_2(j)*Y_1(i));
                end
            end
            
            delta_1 =  (delta_2 * w_2(:,2:end)) .* ( (1-sigmf(I_1',[0.2 10])) .* (sigmf(I_1',[0.2 10])) );
            for j = 1:size(delta_1',1)
                for i = 1:size(amostra,2)
                    w_1(j,i) = w_1(j,i) + (taxaApresendizagem*delta_1(j)*amostra(i));
                end
            end
        end
        
        %Obter Y_2 ajustado
        for a = 1:size(f_features,1)
            amostra = f_features(a,:);
            amostra = mat2gray(amostra);
            %Passo Forward
            for n1 = 1:size(I_1,1)
                I_1(n1) = sum(amostra.*w_1(n1,:));
                Y_1(n1+1) = sigmf( I_1(n1),[0.2 10] );
            end
            for n2 = 1:size(I_2,1)
                I_2(n2) = sum(w_2(n2,:).*Y_1');
                %Y_2(n2) = sigmf(I_2(n2),[0.4 15]);
                Y_2(n2) = sigmf(I_2(n2),[0.2 10]); 
            end
            SAIDA = zeros(1,10);
            SAIDA(max(Y_2)==Y_2)= 1;
            outputs(a,:) = SAIDA;
        end
        erroAtual = E(label,outputs);
        %if erroAtual < erros(epocas)
        %    save('VARS.mat','w_1','w_2','erroAtual');
        %end
        epocas = epocas + 1;
        toc
    end
    erros = [erros,erroAtual];
    %plotando curva de erro
    %plot(1:epocas+1,erros,'Marker','.','LineStyle','-');
    %xlabel('Épocas','FontSize',16);
    %ylabel({'Erro'},'FontSize',16);
    %%Evaluate Base Testing
toc    
end

function [output] = E(label,outputs)
    erros = zeros(size(outputs,1),1);
    for k =1 :size(outputs,1)
        if sum(label(k,:) == outputs(k,:))==10
            erros(k) = 0;
        else
            erros(k) = 1;
        end
    end
    erro = sum(erros.^2)/2;
    output = erro/size(outputs,1);
end