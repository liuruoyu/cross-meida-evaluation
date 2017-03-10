function [O] = getobservations2(nrow, ncol, p)
    O = spalloc(nrow,ncol,floor(nrow*ncol*p)+ncol);
    nprocessed = 0;
    blocksize = 1000;
    while nprocessed < ncol
        if nprocessed+blocksize >= ncol
            currentidx = nprocessed+1:ncol;
            nprocessed = ncol;
        else
            currentidx = nprocessed+1:nprocessed+blocksize;
            nprocessed = nprocessed+blocksize;
        end
        OV = rand(nrow, length(currentidx));
        O(:,currentidx) = sparse(bsxfun(@le, OV, p));
        fprintf('No. of processed: %d\n', nprocessed);
    end
    clear OV
end