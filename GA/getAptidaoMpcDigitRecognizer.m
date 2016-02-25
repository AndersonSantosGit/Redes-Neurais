function [aptidao] = getAptidaoMpcDigitRecognizer(w_1,w_2,data,label,n_intermediaria)

    %Entradas para os neurônios
    I_1 = zeros(n_intermediaria,1);
    I_2 = zeros(10,1);
    
    %Saida dos Neurônios
    Y_1 = ones(n_intermediaria+1,1).*-1;
    Y_2 = ones(10,1).*-1;
    
    outputs = zeros(size(data,1),10);
    for a = 1:size(data,1)
        amostra = data(a,:);
        amostra = mat2gray(amostra);
        %Passo Forward
            for n1 = 1:size(I_1,1)
                I_1(n1) = sum(amostra.*w_1(n1,:));
                Y_1(n1+1) = sigmf( I_1(n1),[0.2 10] );
            end
            for n2 = 1:size(I_2,1)
                I_2(n2) = sum(w_2(n2,:).*Y_1');
                Y_2(n2) = sigmf(I_2(n2),[0.2 10]); 
            end
            SAIDA = zeros(1,10);
            SAIDA(max(Y_2)==Y_2)= 1;
            outputs(a,:) = SAIDA;
    end
    aptidao = getErro(label,outputs);
    
end

function [erros] = getErro(label,outputs)
    erros = 0;
    for k =1 :size(outputs,1)
        if sum(label(k,:) == outputs(k,:))~=10
            erros = erros + 1;
        end
    end
end
