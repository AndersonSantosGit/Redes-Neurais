function [] = pcaDigitRecognizer()
    load('..\PMC\DataSet\Training\Train.mat');
    data  = train(1:100,2:end)';
    [m,n] = size(data);
    k = 3;
    output = zeros(k,n);
    media = mean(data,2);
    C = zeros(m,m);
    E = zeros(m,k);
    %C = mean( (data - media) * (data - media)' );
    for i = 1:n    
        C(i,:) = mean((data(:,i) - media) * (data(:,i) - media)' );
    end
    diagonal = diag(C);
    [~,I] = sort(diagonal);
    output = data(I(1:k),:);
    
end