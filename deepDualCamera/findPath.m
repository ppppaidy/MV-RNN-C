function fullPath = findPath(thisTree,indicies)

elem1 = indicies(1);
elem2 = indicies(2);


parent = thisTree.pp(elem1);
path1 = [];
while parent ~= 0
    path1 = [path1; parent];
    parent = thisTree.pp(parent);
end
path1 = [path1;0];

path2 = [];
parent = thisTree.pp(elem2);
while ~any(parent==path1)
    path2 = [path2; parent];
    parent = thisTree.pp(parent);
end

fullPath = [path1(1:find(path1==parent)); path2(end:-1:1)];

