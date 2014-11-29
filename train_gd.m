function [erms,w] = train_gd(training_data,m_gd,lambda_gd)

    target = training_data(:,1);
    feature = training_data(:,2:end);
    
    % variables
    D = length(feature(1,:));
    M = m_gd;
    N = length(feature(:,1));
    lambda = lambda_gd;
    threshold = 0.001; % threshold for stopping gradient descent
    eta = 1.0; % learning rate
    
    
    % calculate means and variances for basis function
    % partition the feature matrix into M partitions
    % find a random column in each partition
    % find the mean and variance of that column, 
    % this will serve as mean and variance for j-th basis function
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
    
    %save mu_gd.mat means;
    %save s_gd.mat variances;
    
    
    w = zeros([1 M*D+1])';
    i = 1; % counter for i-th data point
    err = 20;
    
    while(i<=N)
        % calculate phi for i-th data point
        phi = ones([M*D+1 1]);
        k = 1; p = 1;
        for j = 2:(M*D+1)
            phi(j) = exp(-((feature(i,k)-means(p))^2)/(2*variances(p)));
            k = k + 1;
            if(k>D)
                k = 1;
                p = p + 1;
            end
        end
        
        %new_w = w + eta * (target(i) - w'*phi) * phi;
        
        % regularized expression for W
        new_w = w * (1 - eta*lambda) - eta * (w'*phi - target(i)) * phi;
        
        new_err = (new_w'*phi - target(i));
        
        if(new_err<err)
            %fprintf('%d\n',i);
            w = new_w;
            if(err-new_err<threshold)
                break;
            end
            err = new_err;
        else
            eta = 0.5 * eta;
        end
        
        i = i + 1;
    end
    
    %save W_gd.mat w;
    
    % calculate phi for whole data
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
    
    error = (phi * w - target)' * (phi * w - target);
    erms = sqrt(2*error/N);
    
end