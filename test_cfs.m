function [erms] = test_cfs(test_data,m_cfs,lambda_cfs,w_cfs)
    
    target = test_data(:,1);
    feature = test_data(:,2:end);
    
    % variables
    D = length(feature(1,:));
    M = m_cfs;
    N = length(feature(:,1));
    lambda = lambda_cfs;
    
    means = zeros([1 M]);
    variances = zeros([1 M]);
    
    partition_begin = 1; partition_end = partition_begin + floor(D/M);
    for j = 1:M
        randomcol_index = floor((partition_end-partition_begin).*rand(1) + partition_begin);
        means(j) = mean(mean(feature(:,randomcol_index)));
        % add small offset to variance to prevent Inf error in basis
        % function
        variances(j) = mean(var(feature(:,randomcol_index))) + 0.0001;
        partition_begin = partition_end; 
        partition_end = partition_begin + floor(D/M);
        if(partition_end > D)
            partition_end = partition_begin + (D - partition_begin);
        end
    end
    
    % calculate the design matrix phi
    phi = zeros([N M*D+1]);
    phi(:,1) = 1; % phi0 = 1
    k = 1; % k-th feature dimension
    p = 1; % p-th basis function
    
    for i = 1:N
       for j = 2:(M*D+1)
          % calculate phi using gaussian basis function
          phi(i,j) = exp(-((feature(i,k) - means(p))^2)/(2*variances(p)));
          k = k + 1;
          if(k>D)
              k = 1;
              p = p + 1;
          end
       end
       p = 1;
    end
    
    I = eye(length(phi(1,:)));
    w_cfs = (phi' * phi + lambda * I)\(phi' * target);

    error = (phi * w_cfs - target)' * (phi * w_cfs - target);
    erms = sqrt(2*error/N);
    
end