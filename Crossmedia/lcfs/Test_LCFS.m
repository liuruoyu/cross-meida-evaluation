%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description
%  evaluation

%Input
%  Test_a     n*dim_a data matrix from one space (e.g. the image space)
%  Test_b     n*dim_b data matrix from another space (e.g. the text space)
%  test_Y     n*1 label vector
%  W_a        dim_a*c projection matrix
%  W_b        dim_b*c projection matrix

%Output
%  map1   MAP score of Test_a as query
%  map2   MAP score of Test_b as query
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  [ map1, map2 ] = Test_LCFS(Test_a, Test_b, test_Y, W_a, W_b)
    
    % the projected data
    data_a = Test_a * W_a;
    data_b = Test_b * W_b;
    
     
    map1 = calculateMAP( data_a, data_b, test_Y );
    str = sprintf( 'The MAP of image as query is %f%%\n', map1 *100 );
    disp(str);
    
    map2 = calculateMAP( data_b, data_a, test_Y );
    str = sprintf( 'The MAP of text as query is %f%%\n', map2 *100 );
    disp(str);    
end

