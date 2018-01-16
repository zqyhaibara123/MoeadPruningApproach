function subp=init_weights(popsize, niche, objDim)
% init_weights function initialize a pupulation of subproblems structure
% with the generated decomposition weight and the neighbourhood
% relationship.
    subp=[];
    if objDim==2
        for i=1:popsize
            p=struct('weight',[],'neighbour',[],'optimal', Inf, 'optpoint',[], 'curpoint', []);
            weight=zeros(2,1);
            weight(1)=i/popsize;
            weight(2)=(popsize-i)/popsize;
            p.weight=weight;
            subp=[subp p];
        end
    elseif objDim==3
        subpar=10;
        for i=1:subpar%popsize=300,if popsize=50?
            for j=1:subpar
                if (i+j<=subpar+1)
                    p=struct('weight',[],'neighbour',[],'optimal', Inf, 'optpoint',[], 'curpoint', []);
                    weight=zeros(3,1);
                    weight (1)=i/subpar;
                    weight(2)=j/subpar;
                    weight(3)=(subpar+2-i-j)/subpar;
                    p.weight=weight;
                    subp=[subp p];
                end
            end
        end
	subp =subp(6:end);
    end
    
% weight = lhsdesign(popsize, objDim, 'criterion','maximin', 'iterations', 1000)';
% p=struct('weight',[],'neighbour',[],'optimal', Inf, 'optpoint',[], 'curpoint', []);
% subp = repmat(p, popsize, 1);
% cells = num2cell(weight);
% [subp.weight]=cells{:};
    
    %Set up the neighbourhood.
    leng=length(subp);
    distanceMatrix=zeros(leng, leng);
    for i=1:leng
        for j=i+1:leng
            A=subp(i).weight;B=subp(j).weight;
            distanceMatrix(i,j)=(A-B)'*(A-B);
            distanceMatrix(j,i)=distanceMatrix(i,j);
        end
        [s,sindex]=sort(distanceMatrix(i,:));
        subp(i).neighbour=sindex(1:niche)';
    end
   
end
