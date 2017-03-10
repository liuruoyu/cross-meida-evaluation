function [Q,C,info] = retrieval(q,gt_q,retriev,gt_r,options)
%
%	[Q,C,info] = retrieval(quer,gt_q,retr,gt_r,opt)
%
%		quer 	: query set (one sample per row)
%		gt_q	: groundtruth for the query set
%
%		retr	: retrieval set (one sample per row)
%		gt_r	: groundtruth for the retrieval set
%
%		opt     : default options for retrieval
%			opt.k = 10 		Precision@k
%			opt.metric = 'L2'	metric for distance calculation
%						'L1', 'L2', 'KL', 'NC' or 'CC'
%			opt.remove = 1 		(default=0) set to 1 removes diagonal from 
%						distance mtx 
%						NOTE: remove is useful if: quer == retr
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	if nargin < 4,
		assert(0==1,'Too few arguments. Type "help retrieval".');
	end;
	save_pwd = pwd;
	tmpdir = '/var/tmp/';

	% default opt -------------
        PREC_K=10;
	METRIC='L2';
	REM_DIAG=0;
	%--------------------------
	if nargin == 5,
		if isfield(options,'metric'), METRIC  =options.metric; end
		if isfield(options,'rm'),     REM_DIAG=options.rm; end
		if isfield(options,'rem'),    REM_DIAG=options.rem; end
		if isfield(options,'remove'), REM_DIAG=options.remove; end
		if isfield(options,'k'),      PREC_K  =options.k; end
	end;
	%--------------------------


	distAll = calc_distance(q+eps,retriev+eps,METRIC);
	dist_lower_dim=min(size(distAll));
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% compute IR metrics retrieval
	%
	%  ** diagonal of distance matrix = Inf **
	%  remove diagonal distances if options.diag=1
	if (REM_DIAG),
		for s = 1:dist_lower_dim,
	       	    distAll(s,s)=inf;
	       	end;
	end

	[query,class] = ir_perquery2(gt_q,distAll(:,:)+1, gt_r);
	Q=query;
	C=class;
	% mAP / P@k / R-Precision  
	M_map = query.map;
	M_prec_at_k  = mean(query.pn(:,PREC_K));
	M_rprecision = query.rprecision;
	% Avg. Precision
	[pr,all]=getprfrompn(query.pn,1);
	M_ap=mean(pr);
	% 11pt interpolated Avg. Precision
	[pr,all]=getprfrompn(query.pn,0);
	M_11ap=mean(pr);
	
	info.mAP=M_map;
	info.mAP_per_class=C.map_class';
	info.Precision_at_k=M_prec_at_k;
	info.k=PREC_K;
	info.RP=M_rprecision;
	info.metric=METRIC;
        info.dist=distAll;
	info.ap_11pt=M_11ap;
	info.ap=M_ap;

