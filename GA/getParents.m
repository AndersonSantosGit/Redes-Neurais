function [parents, maisApto, maiorAptidao] = getParents(w_1,w_2,f_features,label,population, numPais)
    
    roletaData = [];    
    maiorAptidao = 0;
    aptidoes = zeros(size(population, 1),1);
    for i = 1:size(population, 1)
       %tic
       disp([ num2str(i) '>>>' num2str(size(population, 1))]);
       %Obtendo a aptidão - FUNÇÃO DE APTIDÃO   w_1,w_2,f_features,label
       aptidoes(i) = getAptidaoMpcDigitRecognizer(w_1,w_2,f_features,label,population(i, :), todosIndividuos);
       dados = repmat(i, 1, aptidoes(i));
       roletaData = [roletaData, dados];
       
       if aptidoes(i) > maiorAptidao
           maiorAptidao = aptidoes(i);
           maisApto = population(i, :);           
       end
       %toc
    end
    
    parents = roleta(roletaData, population, numPais);        
end

function parents = roleta(roletaData, population, numPais)

    roletaData = shuffle(roletaData, 2);
    indices = randi(size(roletaData, 2),1,size(population,1));
    parents = [];
    
    for i = 1:numPais
        parents = [parents; population(roletaData(indices(i)),:)];        
    end

end