function fuckyou = outputCsv(filename, a, outputtype)
	fid=fopen(filename,'w');
	
	if outputtype == 0
		fprintf(fid,'%d\n',length(a));
		for i=1:length(a)
			fprintf(fid,'%d\n',length(a{i}));
			for j=1:length(a{i})
				fprintf(fid,'%d ',a{i}(1,j));
			end
			fprintf(fid,'\n');
		end
	end
	
	if outputtype == 1
		fprintf(fid,'%d\n',length(a));
		for i=1:length(a)
			fprintf(fid,'%d\n',length(a{i}));
			for j=1:length(a{i})
				fprintf(fid,'%s ',cell2mat(a{i}(1,j)));
			end
			fprintf(fid,'\n');
		end
	end
	
	if outputtype == 2
		fprintf(fid,'%d\n',length(a));
		for i=1:length(a)
			[n m] = size(a{i});
			fprintf(fid,'%d %d\n',n,m);
			for j=1:n
				for k=1:m
					fprintf(fid,'%d ',a{i}(j,k));
				end
				fprintf(fid,'\n');
			end
			fprintf(fid,'\n');
		end
	end
	
	if outputtype == 3
		[n m] = size(a);
		fprintf(fid,'%d %d\n',n,m);
		for i=1:n
			for j=1:m
				fprintf(fid,'%d ',a(i,j));
			end
			fprintf(fid,'\n');
		end
	end
	
	if outputtype == 4
		[n m] = size(a);
		fprintf(fid,'%d %d\n',n,m);
		for i=1:n
			for j=1:m
				fprintf(fid,'%s ',cell2mat(a(i,j)));
			end
			fprintf(fid,'\n');
		end
	end
	
	if outputtype == 5
		[n m] = size(a);
		fprintf(fid,'%d %d\n',n,m);
		for i=1:n
			for j=1:m
				fprintf(fid,'%.15f ',a(i,j));
			end
			fprintf(fid,'\n');
		end
	end
	
	fclose(fid);
return
