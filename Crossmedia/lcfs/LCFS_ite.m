%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description
%  Learning Coupled Feature Spaces

%Input
%  Data_a     n*dim_a data matrix from one space (e.g. the image space)
%  Data_b     n*dim_b data matrix from another space (e.g. the text space)
%  Label      n*1 label vector
%  lambda_1   regularization parameter of L2,1 norm
%  lambda_2   regularization parameter of trace norm
%  ite        the number of iteration 

%Output
%  W_a      dim_a*c projection matrix
%  W_b      dim_b*c projection matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ W_a, W_b ] = LCFS_ite( Data_a, Data_b, Label, lambda_1, lambda_2, ite  )
    tic;
    
    Y = SY2MY( Label );
    Y(find(Y==-1))=0;
    
    [num, dim_a] = size( Data_a );
    [num, dim_b] = size( Data_b );
    [num, c] = size( Y );
    
    % set initial U_a, U_b
    U_a = eye(dim_a, c);
    U_b = eye(dim_b, c);
    
    ppa = ones(dim_a, 1);
    ppb = ones(dim_b, 1);
    %ite = 5;
    
    for i = 1:ite
        % calculate P_a, P_b
        P_a = diag( ppa );
        P_b = diag( ppb );
        
        % calculate inv(S)
        Temp = Data_a * U_a * U_a' * Data_a' + Data_b * U_b * U_b' * Data_b';
        Temp = max(Temp, Temp');
        [V D] = eig( Temp );
        dd = diag( D );
        sqrt_dd = sqrt(dd + 0.0000001);
        dd = 1./sqrt_dd;
        S_inv = V * diag(dd) * V';
        
        % calculate U_a, U_b
        U_a = (Data_a' * Data_a + lambda_1 * P_a + lambda_2 * Data_a' * S_inv * Data_a)\(Data_a' * Y);
        U_b = (Data_b' * Data_b + lambda_1 * P_b + lambda_2 * Data_b' * S_inv * Data_b)\(Data_b' * Y);  
        
        %%%%%%%%%%%%%
        ppa = U_a.*U_a;
        ppa = 1./( 2.*sqrt(sum(ppa,2)+0.0000001) );
        
        ppb = U_b.*U_b;
        ppb = 1./( 2.*sqrt(sum(ppb,2)+0.0000001) );
        disp(i);
    end
    
    W_a = U_a;
    W_b = U_b;
    
    toc;
end

